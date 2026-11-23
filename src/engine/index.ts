import type { Diagnostics, Stroke } from '../capture';
import type { HintLevel } from '../hints/safety';
import type { StallEvent, StallType, Thresholds } from '../stalls';
import {
  ANNOTATION_LABEL_ZH,
  AnnotationTracker,
  type AnnotationLabel,
  type AnnotationRecord,
  type StateTraceSample,
} from './annotationTracker';
import {
  CognitiveStateEngine,
  COMPANION_MODE_LABELS,
  type CognitiveState,
  type CompanionMode,
  type MotorFingerprint,
} from './cognitiveState';
import { EdgeHintModelAdapter } from './edgeHintModel';
import {
  EdgeInkSyntaxEncoder,
  extractSpatialStream,
  getSharedNeuralRunner,
  type EncodedExpressionResult,
} from './inkSyntaxEncoder';
import {
  COGNITIVE_CATEGORIES,
  COGNITIVE_CATEGORY_ZH,
  EdgeNeuralModelRunner,
  type CognitiveCategory,
  type NeuralCognitiveOutput,
  type NeuralMathOutput,
} from './neuralRuntime';
import {
  HINT_STYLES,
  HINT_STYLE_LABELS,
  PersonalizedPolicyEngine,
  type FeedbackLogEntry,
  type FeedbackSignal,
  type HintStyle,
  type PolicySnapshot,
} from './personalizedPolicy';
import {
  parseLatexBlocks,
  ProblemIngestorAdapter,
  type IngestedProblemResult,
  type ProblemContentBlock,
} from './problemIngestor';
import {
  loadPersistedProfile,
  savePersistedProfile,
  type PersistedUserProfile,
} from './profileStore';
import { SAMPLE_PROBLEMS, type SampleProblemSpec } from './sampleProblems';
import {
  getOfflineStyledHint,
  SolutionCacheManager,
  type SolutionCacheEntry,
} from './solutionCache';
import { LocalStorageAdapter } from './storageAdapter';
import type {
  EngineDiagnosticsTick,
  EngineSessionPayload,
  IEdgeHintModelAdapter,
  IHandwritingPulseEngine,
  IInkSyntaxEncoderAdapter,
  IProblemIngestorAdapter,
  IStorageAdapter,
} from './types';

export interface ProblemCacheOverview {
  cached: boolean;
  sourceMode?: SolutionCacheEntry['sourceMode'];
  stepCount: number;
  pitfallCount: number;
  hasSealedAnswer: boolean;
  stepGoals: string[];
}

export {
  ANNOTATION_LABEL_ZH,
  AnnotationTracker,
  COGNITIVE_CATEGORIES,
  COGNITIVE_CATEGORY_ZH,
  COMPANION_MODE_LABELS,
  EdgeHintModelAdapter,
  EdgeInkSyntaxEncoder,
  EdgeNeuralModelRunner,
  extractSpatialStream,
  getSharedNeuralRunner,
  HINT_STYLES,
  HINT_STYLE_LABELS,
  loadPersistedProfile,
  LocalStorageAdapter,
  parseLatexBlocks,
  ProblemIngestorAdapter,
  SAMPLE_PROBLEMS,
  savePersistedProfile,
  type AnnotationLabel,
  type AnnotationRecord,
  type CognitiveCategory,
  type CognitiveState,
  type CompanionMode,
  type EncodedExpressionResult,
  type FeedbackLogEntry,
  type FeedbackSignal,
  type HintStyle,
  type IngestedProblemResult,
  type MotorFingerprint,
  type NeuralCognitiveOutput,
  type NeuralMathOutput,
  type PersistedUserProfile,
  type PolicySnapshot,
  type ProblemContentBlock,
  type SampleProblemSpec,
  type SolutionCacheEntry,
  type StateTraceSample,
};

export type {
  EngineDiagnosticsTick,
  EngineSessionPayload,
  IEdgeHintModelAdapter,
  IHandwritingPulseEngine,
  IInkSyntaxEncoderAdapter,
  IProblemIngestorAdapter,
  IStorageAdapter,
};

interface PendingObservation {
  style: HintStyle;
  context: number[];
  shownAt: number;
  initialMomentumM: number;
  activeResumeStartAt: number | null;
  settled: boolean;
}

const STATE_MACHINE_SUSTAIN_MS = 2000;
const AUTO_RESUME_WINDOW_MS = 5500;
const AUTO_RESUME_SUSTAIN_MS = 450;
const QUICK_DISMISS_MS = 1500;
const THINKING_HOLD_SUPPRESS_MS = 8500;

const NEURAL_WINDOW_STEPS = 20;
const NEURAL_DYNAMICS_DIM = 10;

interface StrokeStepFeature {
  strokeId: number;
  endT: number;
  dyn10: Float32Array;
  sem64: Float32Array;
}

/**
 * 《笔迹脉搏》端侧认知引擎核心调度器（完整四层双流神经状态机架构协调中枢）。
 *
 * 【自研实现】整合：
 * 1. 双流神经网络：EdgeMathEncoder 量化端侧公式识别 + CognitiveStateNet 严格因果 6 分类认知状态推断；
 * 2. 多源观测：笔迹流语法编码器 (EdgeInkSyntaxEncoder) + 解题图谱缓存 (SolutionCacheManager) + 题目摄入器 (ProblemIngestor)；
 * 3. 内环个性化：个人运动指纹在线分布自适应 (MotorFingerprint EMA)；
 * 4. 外环个性化：7 维 LinUCB 六风格提示路由 + L0-L3 渐进提示 + 5 秒恢复书写隐式闭环 + 伴随式零负担微标注。
 */
export class HandwritingPulseEngine implements IHandwritingPulseEngine {
  private readonly cognition = new CognitiveStateEngine();
  private readonly policy = new PersonalizedPolicyEngine();
  private readonly solutionCache = new SolutionCacheManager();
  private readonly neuralRunner = getSharedNeuralRunner();
  private readonly syntaxEncoder = new EdgeInkSyntaxEncoder();
  private readonly problemIngestor = new ProblemIngestorAdapter();
  private readonly builtInEdgeAdapter = new EdgeHintModelAdapter();
  private readonly annotationTracker = new AnnotationTracker();
  private readonly storage = new LocalStorageAdapter();

  private customEdgeAdapter: IEdgeHintModelAdapter | null = null;
  private problemId = '';
  private problemText = '';
  private activeStrokes: Stroke[] = [];
  private latestSyntax: EncodedExpressionResult | null = null;
  private strokeSemCache = new Map<number, Float32Array>();
  private stepHistory: StrokeStepFeature[] = [];

  private nextCognitiveEventId = 10001;
  private lastCognitiveStallAt = -Infinity;
  private lastPitfallTaggedKey = '';
  private thinkingHoldUntil = 0;
  private stateTriggerStartedAt: number | null = null;
  private lastSpeedRatio = 1;
  private pendingObs: PendingObservation | null = null;
  private latestSelection: { style: HintStyle; context: number[] } | null = null;

  constructor() {
    this.syntaxEncoder.setNeuralRunner(this.neuralRunner);

    // 预装离线典型例题解题图谱，保证断网状态也可直接体验步骤对齐与歧路预警
    for (const sample of SAMPLE_PROBLEMS) {
      this.solutionCache.preloadEntry(sample.cacheEntry);
    }

    this.builtInEdgeAdapter.bindContextProvider(() => ({
      solutionEntry: this.solutionCache.getEntry(this.problemText),
      matchedStepIndex: this.latestSyntax?.matchedStepIndex ?? -1,
      hitPitfall: this.latestSyntax?.hitPitfall ?? null,
      recognizedLatex: this.latestSyntax?.recognizedLatex ?? '',
    }));
  }

  public async bootstrapStorage(): Promise<{
    fingerprint: MotorFingerprint;
    policy: PolicySnapshot;
  } | null> {
    const cachedList = await this.storage.loadSolutionCache();
    this.solutionCache.hydrateEntries(cachedList);

    const profile = await this.storage.loadUserProfile('default');
    if (profile) {
      this.hydrateProfile(profile);
    }
    return profile;
  }

  public hydrateProfile(saved: {
    fingerprint: MotorFingerprint;
    policy: PolicySnapshot;
  } | null): void {
    if (!saved) return;
    this.cognition.hydrateFingerprint(saved.fingerprint);
    this.policy.hydrate(saved.policy);
    this.cognition.setCompanionMode(this.policy.getMode());
  }

  public exportPersistedProfile(): {
    fingerprint: MotorFingerprint;
    policy: PolicySnapshot;
  } {
    return {
      fingerprint: this.cognition.getFingerprint(),
      policy: this.policy.exportSnapshot(),
    };
  }

  public async persistAllState(): Promise<void> {
    await this.storage.saveUserProfile('default', this.exportPersistedProfile());
    await this.storage.saveSolutionCache(this.solutionCache.exportAllEntries());
  }

  public resetPersonalization(): void {
    this.cognition.resetFingerprint();
    this.policy.reset();
    this.pendingObs = null;
    this.thinkingHoldUntil = 0;
  }

  public resetSessionTracking(): void {
    this.activeStrokes = [];
    this.latestSyntax = null;
    this.strokeSemCache.clear();
    this.stepHistory = [];
    this.lastPitfallTaggedKey = '';
    this.thinkingHoldUntil = 0;
    this.cognition.resetSessionState();
    this.annotationTracker.reset();
  }

  public setCompanionMode(mode: CompanionMode): void {
    this.cognition.setCompanionMode(mode);
    this.policy.setMode(mode);
  }

  public getCompanionMode(): CompanionMode {
    return this.policy.getMode();
  }

  public getPolicySnapshot(): PolicySnapshot {
    return this.policy.exportSnapshot();
  }

  public getCognitiveState(): CognitiveState {
    return this.cognition.getState();
  }

  public getLatestSyntaxResult(): EncodedExpressionResult | null {
    return this.latestSyntax;
  }

  public getAnnotationCounts(): Record<AnnotationLabel, number> {
    return this.annotationTracker.getCountsByLabel();
  }

  public getSyntaxContextForHint(): {
    recognizedLatex?: string;
    matchedStepIndex?: number;
    pitfallWarning?: string;
  } {
    return {
      ...(this.latestSyntax?.recognizedLatex
        ? { recognizedLatex: this.latestSyntax.recognizedLatex }
        : {}),
      ...(this.latestSyntax && this.latestSyntax.matchedStepIndex >= 0
        ? { matchedStepIndex: this.latestSyntax.matchedStepIndex }
        : {}),
      ...(this.latestSyntax?.hitPitfall?.warningHint
        ? { pitfallWarning: this.latestSyntax.hitPitfall.warningHint }
        : {}),
    };
  }

  public setSemanticAlignment(alignmentA: number): void {
    this.cognition.setSemanticAlignment(alignmentA);
  }

  public registerEdgeModelAdapter(adapter: IEdgeHintModelAdapter | null): void {
    this.customEdgeAdapter = adapter;
  }

  public registerSyntaxEncoderAdapter(adapter: IInkSyntaxEncoderAdapter | null): void {
    this.syntaxEncoder.setNeuralEncoder(adapter);
  }

  public setProblemTextOnly(problemText: string): void {
    this.problemText = problemText;
  }

  public async ingestProblemDraftFromOcr(rawOcrText: string): Promise<IngestedProblemResult> {
    return this.problemIngestor.refineOcrText(rawOcrText);
  }

  public async ingestProblemDraftFromImage(
    imageUri: string,
    hintMlkitText?: string,
  ): Promise<IngestedProblemResult> {
    return this.problemIngestor.ingestFromImageUri(imageUri, hintMlkitText);
  }

  public getProblemCacheOverview(problemText = this.problemText): ProblemCacheOverview {
    const entry = this.solutionCache.getEntry(problemText);
    if (!entry) {
      return {
        cached: false,
        stepCount: 0,
        pitfallCount: 0,
        hasSealedAnswer: false,
        stepGoals: [],
      };
    }
    return {
      cached: true,
      sourceMode: entry.sourceMode,
      stepCount: entry.steps.length,
      pitfallCount: entry.commonPitfalls.length,
      hasSealedAnswer: Boolean(entry.sealedFinalAnswer),
      stepGoals: entry.steps.map(s => s.goalDescription).filter(Boolean).slice(0, 5),
    };
  }

  public async updateProblemText(problemText: string): Promise<boolean> {
    const ingested = await this.problemIngestor.ingestFromText(problemText);
    this.problemId = ingested.problemId;
    this.problemText = ingested.problemText;
    this.lastPitfallTaggedKey = '';

    if (!ingested.problemText) return false;
    if (ingested.matchedSample) {
      this.solutionCache.preloadEntry(ingested.matchedSample.cacheEntry);
      await this.recomputeSyntaxAlignment();
      return true;
    }

    const entry = await this.solutionCache.prewarm(ingested.problemText);
    if (entry) {
      void this.storage.saveSolutionCache(this.solutionCache.exportAllEntries());
    }
    await this.recomputeSyntaxAlignment();
    return entry !== null;
  }

  public async ingestProblemImageUri(imageUri: string): Promise<string> {
    const ingested = await this.problemIngestor.ingestFromImageUri(imageUri);
    await this.updateProblemText(ingested.problemText);
    return ingested.problemText;
  }

  public hasCachedProblem(problemText: string): boolean {
    return this.solutionCache.hasPrewarmed(problemText);
  }

  public async onStrokeComplete(stroke: Stroke): Promise<void> {
    this.activeStrokes.push(stroke);
    if (stroke.tool === 'eraser') {
      this.annotationTracker.recordEraseOrUndo(stroke, 'eraser_event');
    }
    await this.recomputeSyntaxAlignment();
    this.rebuildStepWindowAndInferNeural(stroke.id, this.latestSyntax?.alignmentScore ?? 0);
  }

  public async onStrokeUndo(stroke: Stroke): Promise<void> {
    const idx = this.activeStrokes.findIndex(s => s.id === stroke.id);
    if (idx >= 0) {
      this.activeStrokes.splice(idx, 1);
    } else {
      this.activeStrokes.pop();
    }
    this.strokeSemCache.delete(stroke.id);
    this.annotationTracker.recordEraseOrUndo(stroke, 'undo_event');
    await this.recomputeSyntaxAlignment();
    this.rebuildStepWindowAndInferNeural(-1, this.latestSyntax?.alignmentScore ?? 0);
  }

  private rebuildStepWindowAndInferNeural(newestStrokeId: number, newestAlignA: number): void {
    const prevAlignById = new Map<number, number>();
    for (const prev of this.stepHistory) {
      prevAlignById.set(prev.strokeId, prev.dyn10[9]);
    }

    const fp = this.cognition.getFingerprint();
    const baseSpeed = Math.max(40, 160 * fp.speedRatioEma);
    let prevEndT: number | null = null;
    let runningMomentum = 0.65;
    let decayPause = 0;
    let decayErase = 0;
    const rebuilt: StrokeStepFeature[] = [];

    for (const s of this.activeStrokes) {
      const pts = s.points;
      if (pts.length === 0) continue;

      const isEraser = s.tool === 'eraser' ? 1 : 0;
      const tStart = pts[0].t;
      const tEnd = pts[pts.length - 1].t;
      const durMs = Math.max(16, tEnd - tStart);
      const pauseBefore = prevEndT !== null ? Math.max(0, tStart - prevEndT) : 0;
      const dtSincePrev = pauseBefore + durMs;

      let arcLen = 0;
      const segSpeeds: number[] = [];
      for (let k = 1; k < pts.length; k++) {
        const d = Math.hypot(pts[k].x - pts[k - 1].x, pts[k].y - pts[k - 1].y);
        arcLen += d;
        const dtK = Math.max(4, pts[k].t - pts[k - 1].t) / 1000;
        segSpeeds.push(Math.min(2000, d / dtK));
      }
      const meanSpeed = arcLen / (durMs / 1000);
      const speedRatio = Math.min(2.5, meanSpeed / baseSpeed);
      let speedInstab = 0;
      if (segSpeeds.length >= 2) {
        let sumSeg = 0;
        for (const v of segSpeeds) sumSeg += v;
        const meanSeg = sumSeg / segSpeeds.length;
        let varSeg = 0;
        for (const v of segSpeeds) varSeg += (v - meanSeg) * (v - meanSeg);
        speedInstab = Math.min(2.0, Math.sqrt(varSeg / segSpeeds.length) / Math.max(20, meanSeg));
      }

      const pauseRef = Math.max(pauseBefore, 120);
      const pauseZ =
        pauseBefore > 180
          ? Math.max(
              -2.5,
              Math.min(4.0, (Math.log(pauseRef) - fp.pauseLogMean) / Math.max(0.35, fp.pauseLogStd)),
            )
          : -1.0;

      const isFrag = !isEraser && arcLen < 18 && durMs < 220 ? 1 : 0;
      decayPause = decayPause * Math.exp(-dtSincePrev / 1500) + Math.max(0, pauseZ);
      decayErase = decayErase * Math.exp(-dtSincePrev / 10000) + (isEraser + 0.35 * isFrag);

      const prePauseM = runningMomentum;
      runningMomentum = Math.max(
        0.05,
        Math.min(
          1.0,
          0.55 * runningMomentum + 0.45 * Math.min(1.0, speedRatio) * Math.exp(-pauseBefore / 4500),
        ),
      );
      prevEndT = tEnd;

      const alignA =
        s.id === newestStrokeId
          ? newestAlignA
          : (prevAlignById.get(s.id) ?? newestAlignA);

      const dyn10 = new Float32Array([
        Math.min(1.5, Math.log1p(durMs) / 8.0),
        pauseZ,
        speedRatio,
        speedInstab,
        prePauseM,
        isEraser,
        isFrag,
        Math.tanh(decayPause / 2.5),
        Math.tanh(decayErase / 2.5),
        Math.max(-1.0, Math.min(1.0, alignA)),
      ]);

      let sem64 = this.strokeSemCache.get(s.id);
      if (!sem64) {
        sem64 = this.syntaxEncoder.encodeSingleStrokeSemantic(s);
        this.strokeSemCache.set(s.id, sem64);
      }

      rebuilt.push({ strokeId: s.id, endT: tEnd, dyn10, sem64 });
    }

    this.stepHistory = rebuilt;
    if (this.stepHistory.length < 2 || !this.neuralRunner.isLoaded()) {
      return;
    }

    const semDim = this.neuralRunner.getMathEmbedDim();
    const tail =
      this.stepHistory.length > NEURAL_WINDOW_STEPS
        ? this.stepHistory.slice(-NEURAL_WINDOW_STEPS)
        : this.stepHistory;
    const dynWin = new Float32Array(NEURAL_WINDOW_STEPS * NEURAL_DYNAMICS_DIM);
    const semWin = new Float32Array(NEURAL_WINDOW_STEPS * semDim);
    const padOffset = NEURAL_WINDOW_STEPS - tail.length;
    for (let i = 0; i < tail.length; i++) {
      const slot = padOffset + i;
      dynWin.set(tail[i].dyn10, slot * NEURAL_DYNAMICS_DIM);
      semWin.set(tail[i].sem64, slot * semDim);
    }

    const neuralOut = this.neuralRunner.inferCognitiveWindow(dynWin, semWin, NEURAL_WINDOW_STEPS);
    const eventTime = tail[tail.length - 1].endT || Date.now();
    this.cognition.applyNeuralCognitiveOutput(neuralOut, eventTime);
  }

  private async recomputeSyntaxAlignment(): Promise<void> {
    const entry = this.solutionCache.getEntry(this.problemText);
    const encoded = await this.syntaxEncoder.encodeWithSolution(this.activeStrokes, entry);
    this.latestSyntax = encoded;
    this.cognition.setSyntaxObservation({
      alignmentA: encoded.alignmentScore,
      structuralBoundaryFactor: encoded.structuralBoundaryFactor,
      matchedStepIndex: encoded.matchedStepIndex,
      recognizedLatex: encoded.recognizedLatex,
    });

    if (encoded.hitPitfall && encoded.hitPitfall.triggerPattern !== this.lastPitfallTaggedKey) {
      this.lastPitfallTaggedKey = encoded.hitPitfall.triggerPattern;
      this.annotationTracker.addAnnotation({
        timestamp: Date.now(),
        label: 'PITFALL_TRIGGERED',
        source: 'syntax_pitfall',
        note: encoded.hitPitfall.triggerPattern,
      });
    }
  }

  /**
   * 伴随式零负担微标注交互入口：
   * - `DEEP_THINKING_HOLD`：记录深度思考负样本并暂时静默 8.5 秒防打扰；
   * - `STUCK_CONCEPT_GAP` / `STUCK_CALCULATION`：记录卡壳强正样本并产出对应风格的即时触发事件。
   */
  public recordCompanionAnnotation(
    label: 'DEEP_THINKING_HOLD' | 'STUCK_CONCEPT_GAP' | 'STUCK_CALCULATION',
    nowMs = Date.now(),
  ): {
    record: AnnotationRecord;
    preferredStyle?: HintStyle;
    triggerEvent?: StallEvent;
  } {
    const record = this.annotationTracker.addAnnotation({
      timestamp: nowMs,
      label,
      source: 'user_tap_button',
    });

    if (label === 'DEEP_THINKING_HOLD') {
      this.thinkingHoldUntil = nowMs + THINKING_HOLD_SUPPRESS_MS;
      this.stateTriggerStartedAt = null;
      this.recordExplicitFeedback('thinking', nowMs);
      return { record };
    }

    const preferredStyle: HintStyle =
      label === 'STUCK_CONCEPT_GAP' ? 'socratic_question' : 'algebra_check';
    const triggerEvent: StallEvent = {
      id: this.nextCognitiveEventId++,
      type: 'cognitive_stall',
      startedAt: nowMs,
      x: 0,
      y: 0,
      durationMs: 0,
    };
    return { record, preferredStyle, triggerEvent };
  }

  public processDiagnosticsTick(
    diagnostics: Diagnostics,
    thresholds: Thresholds,
    nowMs = Date.now(),
  ): EngineDiagnosticsTick {
    if (diagnostics.speedRatio !== null) {
      this.lastSpeedRatio = diagnostics.speedRatio;
    }

    const cognitiveState = this.cognition.updateFromDiagnostics(nowMs, diagnostics, thresholds);
    const interventionThreshold = this.policy.getInterventionThreshold();

    // 1. 隐式反馈闭环：观察提示后 5 秒内是否恢复主动书写
    const settledFeedback = this.observeImplicitRecovery(
      nowMs,
      diagnostics,
      cognitiveState.momentumM,
    );
    if (settledFeedback?.signal === 'auto_resumed_writing') {
      this.annotationTracker.addAnnotation({
        timestamp: nowMs,
        label: 'HINT_EFFECTIVE',
        source: 'implicit_recovery',
        note: settledFeedback.style,
      });
    }

    const activelyWriting = diagnostics.rate >= 2 && diagnostics.pauseMs < 450;

    // 2. 记录自适应降采样轨迹与自监督顺畅样本
    this.annotationTracker.recordDiagnosticsSample(
      nowMs,
      cognitiveState,
      this.latestSyntax?.matchedStepIndex ?? -1,
      activelyWriting,
    );

    // 3. 状态机自主介入通道（个人分位数 + 逻辑边界异常 + 擦除犹豫抖动 + 歧路漂移多门控联合判定）
    let cognitiveStallEvent: StallEvent | null = null;
    const cooldownOk =
      diagnostics.cooldownMs <= 0 &&
      nowMs - this.lastCognitiveStallAt >= thresholds.cooldownMs &&
      nowMs >= this.thinkingHoldUntil;

    const minPauseGuard = diagnostics.pauseMs >= 1600;
    const boundaryAnomaly =
      minPauseGuard &&
      cognitiveState.pauseZScore >= 1.6 &&
      cognitiveState.structuralBoundaryFactor >= 1.2 &&
      cognitiveState.momentumM <= 0.52;
    const severePersonalStall =
      minPauseGuard &&
      cognitiveState.pauseZScore >= 2.25 &&
      (cognitiveState.prePauseMomentum <= 0.58 || diagnostics.pauseMs >= thresholds.stallMs + 1500);
    const hesitantEraseStall =
      cognitiveState.eraseStress >= 0.45 &&
      (cognitiveState.speedInstability >= 0.14 || cognitiveState.resistanceR >= 0.5);
    const pitfallDrift =
      cognitiveState.alignmentA <= -0.45 ||
      cognitiveState.cognitiveCategory === 'DIRECTION_DRIFT_PITFALL';
    const thinkingHoldSuppressed =
      cognitiveState.cognitiveCategory === 'DEEP_THINKING_HOLD' &&
      !pitfallDrift &&
      cognitiveState.eraseStress < 0.35;
    const neuralMicroHesitation =
      cognitiveState.cognitiveCategory === 'CALC_MICRO_HESITATION' && diagnostics.pauseMs >= 1350;
    const neuralStuck =
      (cognitiveState.cognitiveCategory === 'STUCK_CALCULATION' ||
        cognitiveState.cognitiveCategory === 'STUCK_CONCEPT_GAP') &&
      diagnostics.pauseMs >= 1500;

    const allowTakeover =
      !thinkingHoldSuppressed &&
      (boundaryAnomaly ||
        severePersonalStall ||
        hesitantEraseStall ||
        pitfallDrift ||
        neuralStuck ||
        neuralMicroHesitation);
    const sustainWindowMs = pitfallDrift
      ? 900
      : cognitiveState.cognitiveCategory === 'CALC_MICRO_HESITATION'
        ? 1400
        : STATE_MACHINE_SUSTAIN_MS;
    const suggestedHintLevel: HintLevel =
      cognitiveState.cognitiveCategory === 'CALC_MICRO_HESITATION' && !pitfallDrift ? 0 : 1;

    if (
      cooldownOk &&
      !activelyWriting &&
      allowTakeover &&
      cognitiveState.interventionScore >= interventionThreshold
    ) {
      if (this.stateTriggerStartedAt === null) {
        this.stateTriggerStartedAt = nowMs;
      }
      if (nowMs - this.stateTriggerStartedAt >= sustainWindowMs) {
        this.stateTriggerStartedAt = null;
        this.lastCognitiveStallAt = nowMs;
        cognitiveStallEvent = {
          id: this.nextCognitiveEventId++,
          type: 'cognitive_stall',
          startedAt: nowMs,
          x: 0,
          y: 0,
          durationMs: Math.round(Math.max(diagnostics.pauseMs, sustainWindowMs)),
        };
      }
    } else {
      this.stateTriggerStartedAt = null;
    }

    return {
      cognitiveState,
      interventionThreshold: Number(interventionThreshold.toFixed(2)),
      cognitiveStallEvent,
      suggestedHintLevel,
      settledFeedback,
    };
  }

  public getSuggestedHintLevel(): HintLevel {
    const state = this.cognition.getState();
    return state.cognitiveCategory === 'CALC_MICRO_HESITATION' && state.alignmentA > -0.45 ? 0 : 1;
  }

  private observeImplicitRecovery(
    nowMs: number,
    diagnostics: Diagnostics,
    momentumM: number,
  ): FeedbackLogEntry | null {
    const obs = this.pendingObs;
    if (!obs || obs.settled) return null;

    const elapsed = nowMs - obs.shownAt;
    if (elapsed > AUTO_RESUME_WINDOW_MS) {
      obs.settled = true;
      return null;
    }

    const activelyWriting =
      diagnostics.rate >= 3 && diagnostics.pauseMs < 400 && !diagnostics.rewritten;
    if (elapsed >= 700 && activelyWriting) {
      if (obs.activeResumeStartAt === null) {
        obs.activeResumeStartAt = nowMs;
      }
      const sustained = nowMs - obs.activeResumeStartAt >= AUTO_RESUME_SUSTAIN_MS;
      const momentumRecovered = momentumM - obs.initialMomentumM >= 0.12;
      if (sustained || momentumRecovered) {
        obs.settled = true;
        this.policy.updateFromFeedback({
          timestamp: nowMs,
          style: obs.style,
          context: obs.context,
          signal: 'auto_resumed_writing',
        });
        const logs = this.policy.getFeedbackLogs();
        return logs[logs.length - 1] ?? null;
      }
    } else {
      obs.activeResumeStartAt = null;
    }
    return null;
  }

  public selectStyleForStall(type: StallType, preferredStyle?: HintStyle): HintStyle {
    const state = this.cognition.getState();
    const selection = this.policy.selectStyle({
      momentumM: state.momentumM,
      resistanceR: state.resistanceR,
      alignmentA: state.alignmentA,
      pauseZScore: state.pauseZScore,
      eraseStress: state.eraseStress,
      speedRatio: this.lastSpeedRatio,
      triggerKind: type,
      cognitiveCategory: state.cognitiveCategory,
    });
    const resolvedStyle = preferredStyle ?? selection.style;
    this.latestSelection = { style: resolvedStyle, context: selection.context };
    return resolvedStyle;
  }

  public lookupCachedHint(
    question: string,
    level: HintLevel,
    style: HintStyle,
  ): string | null {
    // 若命中共性易错歧路且为微轻推/首层触发，优先下发精确歧路反问预警
    if ((level === 0 || level === 1) && this.latestSyntax?.hitPitfall?.warningHint) {
      return this.latestSyntax.hitPitfall.warningHint;
    }
    return this.solutionCache.lookupHint(question, level, style)?.text ?? null;
  }

  public cacheOnlineHint(
    question: string,
    level: HintLevel,
    style: HintStyle,
    text: string,
  ): void {
    this.solutionCache.storeHint(question, level, style, text, 'cloud_live');
    void this.storage.saveSolutionCache(this.solutionCache.exportAllEntries());
  }

  public async tryEdgeAdapterHint(params: {
    question: string;
    type: StallType;
    level: HintLevel;
    durationMs: number;
    style: HintStyle;
  }): Promise<string | null> {
    const adapter = this.customEdgeAdapter?.isReady()
      ? this.customEdgeAdapter
      : this.builtInEdgeAdapter.isReady()
        ? this.builtInEdgeAdapter
        : null;
    if (!adapter) return null;
    try {
      const text = await adapter.generateHint(params);
      if (text) {
        this.solutionCache.storeHint(
          params.question,
          params.level,
          params.style,
          text,
          'edge_offline',
        );
      }
      return text || null;
    } catch {
      return null;
    }
  }

  public getOfflineHintText(type: StallType, level: HintLevel, style: HintStyle): string {
    return getOfflineStyledHint(type, level, style);
  }

  public beginHintObservation(style: HintStyle, nowMs = Date.now()): void {
    const state = this.cognition.getState();
    const context =
      this.latestSelection?.style === style
        ? this.latestSelection.context
        : this.policy.buildContextVector({
            momentumM: state.momentumM,
            resistanceR: state.resistanceR,
            alignmentA: state.alignmentA,
            pauseZScore: state.pauseZScore,
            eraseStress: state.eraseStress,
            speedRatio: this.lastSpeedRatio,
            triggerKind: 'pause_erase',
            cognitiveCategory: state.cognitiveCategory,
          });
    this.pendingObs = {
      style,
      context,
      shownAt: nowMs,
      initialMomentumM: state.momentumM,
      activeResumeStartAt: null,
      settled: false,
    };
  }

  public recordExplicitFeedback(
    action: 'request_more' | 'dismiss' | 'thinking',
    nowMs = Date.now(),
  ): FeedbackLogEntry | null {
    const obs = this.pendingObs;
    if (!obs) return null;

    if (action === 'request_more') {
      obs.settled = true;
      this.policy.updateFromFeedback({
        timestamp: nowMs,
        style: obs.style,
        context: obs.context,
        signal: 'request_more',
      });
    } else if (action === 'thinking') {
      obs.settled = true;
      this.policy.updateFromFeedback({
        timestamp: nowMs,
        style: obs.style,
        context: obs.context,
        signal: 'dismiss_thinking',
      });
    } else {
      if (obs.settled) return null;
      obs.settled = true;
      const elapsed = nowMs - obs.shownAt;
      this.policy.updateFromFeedback({
        timestamp: nowMs,
        style: obs.style,
        context: obs.context,
        signal: elapsed <= QUICK_DISMISS_MS ? 'dismiss_quick' : 'dismiss_normal',
      });
    }

    const logs = this.policy.getFeedbackLogs();
    return logs[logs.length - 1] ?? null;
  }

  public getSummary() {
    return {
      problemId: this.problemId || undefined,
      companionMode: this.policy.getMode(),
      interventionThreshold: Number(this.policy.getInterventionThreshold().toFixed(2)),
      fingerprint: this.cognition.getFingerprint(),
      policySnapshot: this.policy.exportSnapshot(),
      latestState: this.cognition.getState(),
      latestSyntax: this.latestSyntax,
      feedbackLogs: this.policy.getFeedbackLogs(),
      annotations: this.annotationTracker.getAnnotations(),
      annotationCounts: this.annotationTracker.getCountsByLabel(),
      stateTrace: this.annotationTracker.getStateTrace(),
      cachedProblems: this.solutionCache.exportAllEntries(),
    };
  }
}
