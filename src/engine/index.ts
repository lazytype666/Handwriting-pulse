import type { Diagnostics } from '../capture';
import type { HintLevel } from '../hints/safety';
import type { StallEvent, StallType, Thresholds } from '../stalls';
import {
  CognitiveStateEngine,
  type CognitiveState,
  type CompanionMode,
  type MotorFingerprint,
} from './cognitiveState';
import {
  PersonalizedPolicyEngine,
  type FeedbackLogEntry,
  type HintStyle,
  type PolicySnapshot,
} from './personalizedPolicy';
import {
  loadPersistedProfile,
  savePersistedProfile,
  type PersistedUserProfile,
} from './profileStore';
import { getOfflineStyledHint, SolutionCacheManager } from './solutionCache';
import type {
  EngineDiagnosticsTick,
  IEdgeHintModelAdapter,
  IHandwritingPulseEngine,
} from './types';

export {
  loadPersistedProfile,
  savePersistedProfile,
  type PersistedUserProfile,
};

export type {
  CognitiveState,
  CompanionMode,
  MotorFingerprint,
} from './cognitiveState';
export type {
  FeedbackLogEntry,
  FeedbackSignal,
  HintStyle,
  PolicySnapshot,
} from './personalizedPolicy';
export {
  HINT_STYLES,
  HINT_STYLE_LABELS,
} from './personalizedPolicy';
export { COMPANION_MODE_LABELS } from './cognitiveState';
export type {
  EngineDiagnosticsTick,
  EngineSessionPayload,
  IEdgeHintModelAdapter,
  IHandwritingPulseEngine,
  IInkSyntaxEncoderAdapter,
  IProblemIngestorAdapter,
  IStorageAdapter,
} from './types';

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

/**
 * 《笔迹脉搏》端侧认知引擎核心调度器。
 *
 * 【自研实现】整合：内环个人运动指纹 + 三维连续认知状态机 [M,R,A] +
 * 外环 LinUCB 四风格策略路由 + 解题图谱缓存 + 5 秒恢复书写隐式反馈闭环。
 */
export class HandwritingPulseEngine implements IHandwritingPulseEngine {
  private readonly cognition = new CognitiveStateEngine();
  private readonly policy = new PersonalizedPolicyEngine();
  private readonly solutionCache = new SolutionCacheManager();

  private edgeAdapter: IEdgeHintModelAdapter | null = null;
  private problemText = '';
  private nextCognitiveEventId = 10001;
  private lastCognitiveStallAt = -Infinity;
  private stateTriggerStartedAt: number | null = null;
  private lastSpeedRatio = 1;
  private pendingObs: PendingObservation | null = null;
  private latestSelection: { style: HintStyle; context: number[] } | null = null;

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

  public resetPersonalization(): void {
    this.cognition.resetFingerprint();
    this.policy.reset();
    this.pendingObs = null;
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

  public setSemanticAlignment(alignmentA: number): void {
    this.cognition.setSemanticAlignment(alignmentA);
  }

  public registerEdgeModelAdapter(adapter: IEdgeHintModelAdapter | null): void {
    this.edgeAdapter = adapter;
  }

  public setProblemTextOnly(problemText: string): void {
    this.problemText = problemText;
  }

  public async updateProblemText(problemText: string): Promise<boolean> {
    this.problemText = problemText;
    if (!problemText.trim()) return false;
    const entry = await this.solutionCache.prewarm(problemText);
    return entry !== null;
  }

  public hasCachedProblem(problemText: string): boolean {
    return this.solutionCache.hasPrewarmed(problemText);
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

    // 2. 状态机自主介入通道（留出 STALL_MS + 2500ms 空间给阶段二规则 A 涂改路径，且遵守冷却）
    let cognitiveStallEvent: StallEvent | null = null;
    const cooldownOk =
      diagnostics.cooldownMs <= 0 &&
      nowMs - this.lastCognitiveStallAt >= thresholds.cooldownMs;

    const activelyWriting = diagnostics.rate >= 2 && diagnostics.pauseMs < 450;
    const allowTakeover =
      diagnostics.pauseMs >= thresholds.stallMs + 2500 ||
      cognitiveState.eraseStress >= 0.55 ||
      cognitiveState.alignmentA <= -0.4;

    if (
      cooldownOk &&
      !activelyWriting &&
      allowTakeover &&
      cognitiveState.interventionScore >= interventionThreshold
    ) {
      if (this.stateTriggerStartedAt === null) {
        this.stateTriggerStartedAt = nowMs;
      }
      if (nowMs - this.stateTriggerStartedAt >= STATE_MACHINE_SUSTAIN_MS) {
        this.stateTriggerStartedAt = null;
        this.lastCognitiveStallAt = nowMs;
        cognitiveStallEvent = {
          id: this.nextCognitiveEventId++,
          type: 'cognitive_stall',
          startedAt: nowMs,
          x: 0,
          y: 0,
          durationMs: Math.round(Math.max(diagnostics.pauseMs, STATE_MACHINE_SUSTAIN_MS)),
        };
      }
    } else {
      this.stateTriggerStartedAt = null;
    }

    return {
      cognitiveState,
      interventionThreshold: Number(interventionThreshold.toFixed(2)),
      cognitiveStallEvent,
      settledFeedback,
    };
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

    const activelyWriting = diagnostics.rate >= 3 && diagnostics.pauseMs < 400 && !diagnostics.rewritten;
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

  public selectStyleForStall(type: StallType): HintStyle {
    const state = this.cognition.getState();
    const selection = this.policy.selectStyle({
      momentumM: state.momentumM,
      resistanceR: state.resistanceR,
      alignmentA: state.alignmentA,
      pauseZScore: state.pauseZScore,
      eraseStress: state.eraseStress,
      speedRatio: this.lastSpeedRatio,
      triggerKind: type,
    });
    this.latestSelection = { style: selection.style, context: selection.context };
    return selection.style;
  }

  public lookupCachedHint(
    question: string,
    level: HintLevel,
    style: HintStyle,
  ): string | null {
    return this.solutionCache.lookupHint(question, level, style)?.text ?? null;
  }

  public cacheOnlineHint(
    question: string,
    level: HintLevel,
    style: HintStyle,
    text: string,
  ): void {
    this.solutionCache.storeHint(question, level, style, text, 'cloud_live');
  }

  public async tryEdgeAdapterHint(params: {
    question: string;
    type: StallType;
    level: HintLevel;
    durationMs: number;
    style: HintStyle;
  }): Promise<string | null> {
    if (!this.edgeAdapter?.isReady()) return null;
    try {
      const text = await this.edgeAdapter.generateHint(params);
      if (text) {
        this.solutionCache.storeHint(params.question, params.level, params.style, text, 'edge_offline');
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
      companionMode: this.policy.getMode(),
      interventionThreshold: Number(this.policy.getInterventionThreshold().toFixed(2)),
      fingerprint: this.cognition.getFingerprint(),
      policySnapshot: this.policy.exportSnapshot(),
      latestState: this.cognition.getState(),
      feedbackLogs: this.policy.getFeedbackLogs(),
      cachedProblems: this.solutionCache.exportAllEntries(),
    };
  }
}
