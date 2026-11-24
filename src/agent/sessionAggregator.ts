import type { Diagnostics, Stroke } from '../capture';
import type { StallEvent } from '../stalls';
import type { CognitiveCategory, CognitiveState } from '../engine/cognitiveState';
import type { FeedbackLogEntry, HintStyle } from '../engine/personalizedPolicy';

export type KnowledgeConceptTag =
  | 'calculus.limit_taylor'
  | 'calculus.integral_bounds'
  | 'calculus.ode'
  | 'linalg.matrix_det'
  | 'linalg.eigen_orthogonal'
  | 'prob.bayes_conditional'
  | 'prob.distribution_norm'
  | 'general.algebra_transform';

export const KNOWLEDGE_TAG_LABELS: Record<KnowledgeConceptTag, string> = {
  'calculus.limit_taylor': '极限与泰勒展开',
  'calculus.integral_bounds': '变限/重积分与代换定限',
  'calculus.ode': '微分方程特征根与通特解',
  'linalg.matrix_det': '含参矩阵行列式与秩讨论',
  'linalg.eigen_orthogonal': '特征值与实对称正交化',
  'prob.bayes_conditional': '全概率与贝叶斯推断',
  'prob.distribution_norm': '随机变量分布与标准化',
  'general.algebra_transform': '代数恒等变形与符号化简',
};

export interface WritingHabitMetrics {
  avgMomentumM: number;
  avgSpeedRatio: number;
  pauseMedianMs: number;
  pauseP90Ms: number;
  pauseZScoreMean: number;
  pauseZScoreP90: number;
  totalEraseCount: number;
  totalUndoCount: number;
  erasePerStrokeRatio: number;
  deepThinkingHoldRatio: number;
  fluentDerivationRatio: number;
  stuckCalculationRatio: number;
  stuckConceptGapRatio: number;
  pitfallDriftRatio: number;
  /** 停顿发生在关键结构边界（= / \frac / \int 等）后的比例，高比例通常反映思维断层而非粗心 */
  structuralBoundaryPauseRatio: number;
}

export interface SessionTimelineMilestone {
  offsetMs: number;
  kind: 'stall' | 'pitfall' | 'step_advance' | 'correction' | 'hint_feedback';
  detail: string;
}

export interface SessionLearningTelemetry {
  schemaVersion: 'agent-telemetry-v1';
  sessionId: string;
  capturedAt: number;
  durationMs: number;
  problemKey: string;
  problemText: string;
  knowledgeTags: KnowledgeConceptTag[];
  totalStrokes: number;
  totalStepsExpected: number;
  maxMatchedStepIndex: number;
  finalStepProgress: number;
  finalAnchorGap: number;
  meanAlignmentA: number;
  maxPitfallProb: number;
  hitPitfalls: string[];
  habits: WritingHabitMetrics;
  hintFeedbackSummary: {
    hintsShown: number;
    acceptedCount: number;
    dismissedCount: number;
    topEffectiveStyle: HintStyle | null;
    rewardSum: number;
  };
  representativeEvents: SessionTimelineMilestone[];
}

export function extractKnowledgeTags(
  problemText: string,
  recognizedLatex = '',
): KnowledgeConceptTag[] {
  const text = `${problemText} ${recognizedLatex}`;
  const tags: KnowledgeConceptTag[] = [];
  if (/\\?lim|极限|泰勒|洛必达|无穷小|未定式/i.test(text)) {
    tags.push('calculus.limit_taylor');
  }
  if (/∫|\\i{1,3}nt|\\oint|变上限|二重积分|积分|分部积分|polar/i.test(text)) {
    tags.push('calculus.integral_bounds');
  }
  if (/微分方程|y''|y'|特解|齐次|通解|\bODE\b|characteristic/i.test(text)) {
    tags.push('calculus.ode');
  }
  if (/矩阵|行列式|秩|Ax\s*=|方程组|\\det|\|A\|/i.test(text)) {
    tags.push('linalg.matrix_det');
  }
  if (/特征值|正交|对称矩阵|相似对角化|\\lambda/i.test(text)) {
    tags.push('linalg.eigen_orthogonal');
  }
  if (/贝叶斯|全概率|条件概率|P\(|后验|\\cap|\\cup/i.test(text)) {
    tags.push('prob.bayes_conditional');
  }
  if (/正态|分布|期望|方差|\\Phi|Φ|均匀分布/i.test(text)) {
    tags.push('prob.distribution_norm');
  }
  if (tags.length === 0) {
    tags.push('general.algebra_transform');
  }
  return tags.slice(0, 3);
}

function percentile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}

/**
 * 端侧单题过程性遥测聚合器（零原始坐标上传）：
 * 在作答全程以有界内存缓冲区采集停顿 Z 分数分布、涂改/撤销节律、图谱步进轨迹与隐式反馈，
 * 并在单题结束或导出时输出高度压缩且隐私友好的 `SessionLearningTelemetry`。
 */
export class SessionTelemetryAggregator {
  private sessionId = `sess_${Date.now()}`;
  private startedAt = Date.now();
  private tickCount = 0;
  private speedRatios: number[] = [];
  private momentums: number[] = [];
  private pauseDurationsMs: number[] = [];
  private pauseZScores: number[] = [];
  private alignments: number[] = [];
  private structuralLongPauses = 0;
  private totalLongPauses = 0;
  private categoryCounts: Record<CognitiveCategory, number> = {
    FLUENT_DERIVATION: 0,
    DEEP_THINKING_HOLD: 0,
    CALC_MICRO_HESITATION: 0,
    STUCK_CALCULATION: 0,
    STUCK_CONCEPT_GAP: 0,
    DIRECTION_DRIFT_PITFALL: 0,
  };
  private maxMatchedStep = -1;
  private maxStepProgress = 0;
  private lastAnchorGap = 0;
  private maxPitfallProb = 0;
  private hitPitfalls = new Set<string>();
  private eraseCount = 0;
  private undoCount = 0;
  private hintsShown = 0;
  private milestones: SessionTimelineMilestone[] = [];
  private lastPauseEpisodeMs = 0;

  public reset(nowMs = Date.now()): void {
    this.sessionId = `sess_${nowMs}_${Math.random().toString(36).slice(2, 7)}`;
    this.startedAt = nowMs;
    this.tickCount = 0;
    this.speedRatios = [];
    this.momentums = [];
    this.pauseDurationsMs = [];
    this.pauseZScores = [];
    this.alignments = [];
    this.structuralLongPauses = 0;
    this.totalLongPauses = 0;
    this.categoryCounts = {
      FLUENT_DERIVATION: 0,
      DEEP_THINKING_HOLD: 0,
      CALC_MICRO_HESITATION: 0,
      STUCK_CALCULATION: 0,
      STUCK_CONCEPT_GAP: 0,
      DIRECTION_DRIFT_PITFALL: 0,
    };
    this.maxMatchedStep = -1;
    this.maxStepProgress = 0;
    this.lastAnchorGap = 0;
    this.maxPitfallProb = 0;
    this.hitPitfalls.clear();
    this.eraseCount = 0;
    this.undoCount = 0;
    this.hintsShown = 0;
    this.milestones = [];
    this.lastPauseEpisodeMs = 0;
  }

  private pushMilestone(nowMs: number, kind: SessionTimelineMilestone['kind'], detail: string): void {
    if (this.milestones.length >= 24) return;
    this.milestones.push({
      offsetMs: Math.max(0, nowMs - this.startedAt),
      kind,
      detail: detail.slice(0, 80),
    });
  }

  private pushBounded(arr: number[], value: number, limit = 256): void {
    if (arr.length >= limit) arr.shift();
    arr.push(value);
  }

  public recordCorrection(kind: 'erase' | 'undo', nowMs = Date.now()): void {
    if (kind === 'erase') this.eraseCount += 1;
    else this.undoCount += 1;
    this.pushMilestone(nowMs, 'correction', kind === 'erase' ? '橡皮涂改重写' : '撤销重来');
  }

  public recordHintShown(nowMs = Date.now()): void {
    this.hintsShown += 1;
    this.pushMilestone(nowMs, 'hint_feedback', `第 ${this.hintsShown} 次触发自适应提示`);
  }

  public recordStepAdvance(stepIndex: number, progress: number, nowMs = Date.now()): void {
    if (stepIndex > this.maxMatchedStep) {
      this.maxMatchedStep = stepIndex;
      this.pushMilestone(nowMs, 'step_advance', `推进至第 ${stepIndex + 1} 步 (进度 ${Math.round(progress * 100)}%)`);
    }
    this.maxStepProgress = Math.max(this.maxStepProgress, progress);
  }

  public recordPitfall(pattern: string, prob: number, nowMs = Date.now()): void {
    this.maxPitfallProb = Math.max(this.maxPitfallProb, prob);
    if (pattern && !this.hitPitfalls.has(pattern)) {
      this.hitPitfalls.add(pattern);
      this.pushMilestone(nowMs, 'pitfall', `触发歧路风险: ${pattern}`);
    }
  }

  public onTick(
    state: CognitiveState,
    diag: Diagnostics,
    stallEvent: StallEvent | null,
    nowMs = Date.now(),
  ): void {
    this.tickCount += 1;
    this.categoryCounts[state.cognitiveCategory] += 1;
    this.pushBounded(this.momentums, state.momentumM);
    this.pushBounded(this.alignments, state.alignmentA);
    this.lastAnchorGap = state.anchorGap;
    this.maxStepProgress = Math.max(this.maxStepProgress, state.stepProgress);
    if (state.matchedStepIndex > this.maxMatchedStep) {
      this.recordStepAdvance(state.matchedStepIndex, state.stepProgress, nowMs);
    }
    if (state.pitfallProb > this.maxPitfallProb) {
      this.maxPitfallProb = state.pitfallProb;
    }

    if (diag.speedRatio !== null && diag.rate > 0) {
      this.pushBounded(this.speedRatios, diag.speedRatio);
    }

    if (diag.pauseMs >= 650) {
      this.lastPauseEpisodeMs = Math.max(this.lastPauseEpisodeMs, diag.pauseMs);
    } else if (this.lastPauseEpisodeMs >= 650) {
      this.pushBounded(this.pauseDurationsMs, this.lastPauseEpisodeMs);
      this.pushBounded(this.pauseZScores, state.pauseZScore);
      this.totalLongPauses += 1;
      if (state.structuralBoundaryFactor >= 1.15) {
        this.structuralLongPauses += 1;
      }
      this.lastPauseEpisodeMs = 0;
    }

    if (stallEvent) {
      this.pushMilestone(
        nowMs,
        'stall',
        `${stallEvent.type} (${state.cognitiveCategory}, 停顿 ${Math.round(stallEvent.durationMs)}ms)`,
      );
    }
  }

  public buildTelemetry(params: {
    problemText: string;
    problemKey: string;
    strokes: Stroke[];
    totalStepsExpected: number;
    feedbackLogs: FeedbackLogEntry[];
    latestState: CognitiveState;
    nowMs?: number;
  }): SessionLearningTelemetry {
    const nowMs = params.nowMs ?? Date.now();
    const sortedPauses = [...this.pauseDurationsMs].sort((a, b) => a - b);
    const sortedZ = [...this.pauseZScores].sort((a, b) => a - b);

    const mean = (xs: number[], fallback: number) =>
      xs.length > 0 ? xs.reduce((a, b) => a + b, 0) / xs.length : fallback;

    const totalTicks = Math.max(1, this.tickCount);
    const penStrokeCount = params.strokes.filter(s => s.tool !== 'eraser').length;

    const styleRewards = new Map<HintStyle, number>();
    let acceptedCount = 0;
    let dismissedCount = 0;
    let rewardSum = 0;
    for (const fb of params.feedbackLogs) {
      rewardSum += fb.reward;
      styleRewards.set(fb.style, (styleRewards.get(fb.style) ?? 0) + fb.reward);
      if (fb.reward > 0) acceptedCount += 1;
      else dismissedCount += 1;
    }
    let topEffectiveStyle: HintStyle | null = null;
    let bestStyleScore = -Infinity;
    for (const [style, score] of styleRewards.entries()) {
      if (score > bestStyleScore) {
        bestStyleScore = score;
        topEffectiveStyle = style;
      }
    }

    return {
      schemaVersion: 'agent-telemetry-v1',
      sessionId: this.sessionId,
      capturedAt: nowMs,
      durationMs: Math.max(0, nowMs - this.startedAt),
      problemKey: params.problemKey,
      problemText: params.problemText,
      knowledgeTags: extractKnowledgeTags(params.problemText, params.latestState.recognizedLatex),
      totalStrokes: penStrokeCount,
      totalStepsExpected: Math.max(1, params.totalStepsExpected),
      maxMatchedStepIndex: Math.max(this.maxMatchedStep, params.latestState.matchedStepIndex),
      finalStepProgress: Number(
        Math.max(this.maxStepProgress, params.latestState.stepProgress).toFixed(3),
      ),
      finalAnchorGap: Number(this.lastAnchorGap.toFixed(3)),
      meanAlignmentA: Number(mean(this.alignments, params.latestState.alignmentA).toFixed(3)),
      maxPitfallProb: Number(Math.max(this.maxPitfallProb, params.latestState.pitfallProb).toFixed(3)),
      hitPitfalls: Array.from(this.hitPitfalls),
      habits: {
        avgMomentumM: Number(mean(this.momentums, params.latestState.momentumM).toFixed(3)),
        avgSpeedRatio: Number(mean(this.speedRatios, 1.0).toFixed(3)),
        pauseMedianMs: Math.round(percentile(sortedPauses, 0.5)),
        pauseP90Ms: Math.round(percentile(sortedPauses, 0.9)),
        pauseZScoreMean: Number(mean(this.pauseZScores, 0).toFixed(2)),
        pauseZScoreP90: Number(percentile(sortedZ, 0.9).toFixed(2)),
        totalEraseCount: this.eraseCount,
        totalUndoCount: this.undoCount,
        erasePerStrokeRatio: Number(
          ((this.eraseCount + this.undoCount) / Math.max(1, penStrokeCount)).toFixed(3),
        ),
        deepThinkingHoldRatio: Number(
          (this.categoryCounts.DEEP_THINKING_HOLD / totalTicks).toFixed(3),
        ),
        fluentDerivationRatio: Number(
          (this.categoryCounts.FLUENT_DERIVATION / totalTicks).toFixed(3),
        ),
        stuckCalculationRatio: Number(
          (
            (this.categoryCounts.STUCK_CALCULATION + this.categoryCounts.CALC_MICRO_HESITATION) /
            totalTicks
          ).toFixed(3),
        ),
        stuckConceptGapRatio: Number(
          (this.categoryCounts.STUCK_CONCEPT_GAP / totalTicks).toFixed(3),
        ),
        pitfallDriftRatio: Number(
          (this.categoryCounts.DIRECTION_DRIFT_PITFALL / totalTicks).toFixed(3),
        ),
        structuralBoundaryPauseRatio: Number(
          (this.structuralLongPauses / Math.max(1, this.totalLongPauses)).toFixed(3),
        ),
      },
      hintFeedbackSummary: {
        hintsShown: this.hintsShown,
        acceptedCount,
        dismissedCount,
        topEffectiveStyle,
        rewardSum: Number(rewardSum.toFixed(2)),
      },
      representativeEvents: [...this.milestones],
    };
  }
}
