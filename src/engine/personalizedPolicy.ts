import type { CompanionMode } from './cognitiveState';

/**
 * 提示风格动作空间（对应外环 Contextual Bandit 的六种大学数学启发动作臂）。
 */
export type HintStyle =
  | 'socratic_question'   // 反问点拨
  | 'condition_reminder'  // 条件提醒
  | 'algebra_check'       // 算式纠错
  | 'step_decompose'      // 拆步引导
  | 'concept_recall'      // 概念唤醒
  | 'sanity_check';       // 特值自检

export const HINT_STYLES: HintStyle[] = [
  'socratic_question',
  'condition_reminder',
  'algebra_check',
  'step_decompose',
  'concept_recall',
  'sanity_check',
];

export const HINT_STYLE_LABELS: Record<HintStyle, string> = {
  socratic_question: '反问点拨',
  condition_reminder: '条件提醒',
  algebra_check: '算式纠错',
  step_decompose: '拆步引导',
  concept_recall: '概念唤醒',
  sanity_check: '特值自检',
};

export type FeedbackSignal =
  | 'auto_resumed_writing' // 提示后 5 秒内顺畅恢复书写 (+1.0)
  | 'request_more'         // 用户点「再提示一点」(+0.6)
  | 'dismiss_normal'       // 正常收起 (-0.1)
  | 'dismiss_quick'        // 1.5 秒内秒关 (-0.7)
  | 'dismiss_thinking';    // 用户点「我在想，别打扰」(-1.0)

export interface FeedbackLogEntry {
  timestamp: number;
  style: HintStyle;
  signal: FeedbackSignal;
  reward: number;
  context: number[];
}

export interface LinUcbArmState {
  aFlat: number[];
  b: number[];
  pulls: number;
  rewardSum: number;
}

export interface PolicySnapshot {
  mode: CompanionMode;
  thresholdOffset: number;
  arms: Record<HintStyle, LinUcbArmState>;
  feedbackCount: number;
}

const CONTEXT_DIM = 7;
const LINUCB_ALPHA = 0.45;

function createIdentityFlat(dim: number): number[] {
  const arr = new Array<number>(dim * dim).fill(0);
  for (let i = 0; i < dim; i++) arr[i * dim + i] = 1;
  return arr;
}

function createInitialArm(): LinUcbArmState {
  return {
    aFlat: createIdentityFlat(CONTEXT_DIM),
    b: new Array<number>(CONTEXT_DIM).fill(0),
    pulls: 0,
    rewardSum: 0,
  };
}

/**
 * 纯 TypeScript 高斯-若尔当矩阵求逆（7×7 正定矩阵耗时 < 0.02ms，零外部依赖）。
 */
function invertMatrix(flat: number[], dim: number): number[] {
  const aug: number[][] = [];
  for (let i = 0; i < dim; i++) {
    const row = new Array<number>(dim * 2).fill(0);
    for (let j = 0; j < dim; j++) row[j] = flat[i * dim + j];
    row[dim + i] = 1;
    aug.push(row);
  }

  for (let col = 0; col < dim; col++) {
    let pivotRow = col;
    let maxVal = Math.abs(aug[col][col]);
    for (let r = col + 1; r < dim; r++) {
      const v = Math.abs(aug[r][col]);
      if (v > maxVal) {
        maxVal = v;
        pivotRow = r;
      }
    }
    if (pivotRow !== col) {
      const tmp = aug[col];
      aug[col] = aug[pivotRow];
      aug[pivotRow] = tmp;
    }
    const pivot = Math.abs(aug[col][col]) < 1e-8 ? 1e-8 : aug[col][col];
    for (let j = 0; j < dim * 2; j++) aug[col][j] /= pivot;

    for (let r = 0; r < dim; r++) {
      if (r === col) continue;
      const factor = aug[r][col];
      if (factor === 0) continue;
      for (let j = 0; j < dim * 2; j++) {
        aug[r][j] -= factor * aug[col][j];
      }
    }
  }

  const inv = new Array<number>(dim * dim);
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      inv[i * dim + j] = aug[i][dim + j];
    }
  }
  return inv;
}

function matVecMul(matFlat: number[], vec: number[], dim: number): number[] {
  const out = new Array<number>(dim).fill(0);
  for (let i = 0; i < dim; i++) {
    let sum = 0;
    const rowOffset = i * dim;
    for (let j = 0; j < dim; j++) sum += matFlat[rowOffset + j] * vec[j];
    out[i] = sum;
  }
  return out;
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function createInitialArms(): Record<HintStyle, LinUcbArmState> {
  const arms = {} as Record<HintStyle, LinUcbArmState>;
  for (const s of HINT_STYLES) {
    arms[s] = createInitialArm();
  }
  return arms;
}

export interface PolicyContextInput {
  momentumM: number;
  resistanceR: number;
  alignmentA: number;
  pauseZScore: number;
  eraseStress: number;
  speedRatio: number;
  triggerKind: 'pause_erase' | 'speed_drop' | 'cognitive_stall';
  cognitiveCategory?: string;
}

/**
 * 外环个性化策略路由器（Disjoint LinUCB 上下文赌博机 + 动态触发门限自适应）。
 *
 * 【自研实现】设计思路：
 * 1. 上下文向量 x ∈ R^7 融合三维认知状态 [M, R, A]、停顿 Z 分数、涂改压力与速度比；
 * 2. 在 6 种启发风格（反问/条件/纠错/拆步/概念唤醒/特值自检）间通过 UCB 权衡探索与利用；
 * 3. 根据隐式反馈（5 秒恢复书写 +1 / 秒关或点“我在想” -1）同步更新风格权重 θ_u 与触发门限偏置。
 */
export class PersonalizedPolicyEngine {
  private mode: CompanionMode = 'socratic';
  private thresholdOffset = 0;
  private arms: Record<HintStyle, LinUcbArmState> = createInitialArms();
  private feedbackLogs: FeedbackLogEntry[] = [];

  public reset(): void {
    this.thresholdOffset = 0;
    this.arms = createInitialArms();
    this.feedbackLogs = [];
  }

  public hydrate(snapshot: PolicySnapshot | null): void {
    if (!snapshot) return;
    this.mode = snapshot.mode ?? 'socratic';
    this.thresholdOffset = clamp(snapshot.thresholdOffset ?? 0, -0.15, 0.25);
    for (const style of HINT_STYLES) {
      const arm = snapshot.arms?.[style];
      if (
        arm &&
        Array.isArray(arm.aFlat) &&
        arm.aFlat.length === CONTEXT_DIM * CONTEXT_DIM &&
        Array.isArray(arm.b) &&
        arm.b.length === CONTEXT_DIM
      ) {
        this.arms[style] = {
          aFlat: [...arm.aFlat],
          b: [...arm.b],
          pulls: arm.pulls ?? 0,
          rewardSum: arm.rewardSum ?? 0,
        };
      }
    }
  }

  public exportSnapshot(): PolicySnapshot {
    const armsCopy = {} as Record<HintStyle, LinUcbArmState>;
    for (const s of HINT_STYLES) {
      armsCopy[s] = { ...this.arms[s], aFlat: [...this.arms[s].aFlat], b: [...this.arms[s].b] };
    }
    return {
      mode: this.mode,
      thresholdOffset: Number(this.thresholdOffset.toFixed(4)),
      arms: armsCopy,
      feedbackCount: this.feedbackLogs.length,
    };
  }

  public getFeedbackLogs(): FeedbackLogEntry[] {
    return this.feedbackLogs;
  }

  public setMode(mode: CompanionMode): void {
    this.mode = mode;
  }

  public getMode(): CompanionMode {
    return this.mode;
  }

  public getInterventionThreshold(): number {
    const base = this.mode === 'zen' ? 0.78 : this.mode === 'coach' ? 0.46 : 0.62;
    return clamp(base + this.thresholdOffset, 0.35, 0.92);
  }

  public buildContextVector(input: PolicyContextInput): number[] {
    return [
      1.0,
      clamp(input.momentumM, 0, 1),
      clamp(input.resistanceR, 0, 1),
      clamp(input.alignmentA, -1, 1),
      clamp(input.pauseZScore / 3, -1, 1),
      clamp(input.eraseStress, 0, 1),
      clamp(input.speedRatio, 0, 1.5),
    ];
  }

  public selectStyle(input: PolicyContextInput): {
    style: HintStyle;
    context: number[];
    scores: Record<HintStyle, number>;
  } {
    const x = this.buildContextVector(input);
    const scores = {} as Record<HintStyle, number>;
    let bestStyle: HintStyle = 'socratic_question';
    let bestScore = -Infinity;

    for (const style of HINT_STYLES) {
      const arm = this.arms[style];
      const aInv = invertMatrix(arm.aFlat, CONTEXT_DIM);
      const theta = matVecMul(aInv, arm.b, CONTEXT_DIM);
      const aInvX = matVecMul(aInv, x, CONTEXT_DIM);
      const uncertainty = Math.sqrt(Math.max(0, dot(x, aInvX)));
      const exploit = dot(theta, x);

      const prior = this.computeColdStartPrior(style, input);
      const priorWeight = Math.max(0.15, 1 / (1 + arm.pulls * 0.25));
      const ucbScore = exploit + LINUCB_ALPHA * uncertainty + prior * priorWeight;

      scores[style] = Number(ucbScore.toFixed(3));
      if (ucbScore > bestScore) {
        bestScore = ucbScore;
        bestStyle = style;
      }
    }

    return { style: bestStyle, context: x, scores };
  }

  private computeColdStartPrior(style: HintStyle, input: PolicyContextInput): number {
    const cat = input.cognitiveCategory;
    if (cat === 'CALC_MICRO_HESITATION' || cat === 'STUCK_CALCULATION') {
      if (style === 'algebra_check') return 0.30;
      if (style === 'sanity_check') return 0.25;
      if (style === 'step_decompose') return 0.20;
      return 0.08;
    }
    if (cat === 'DIRECTION_DRIFT_PITFALL') {
      if (style === 'condition_reminder') return 0.30;
      if (style === 'concept_recall') return 0.26;
      if (style === 'socratic_question') return 0.18;
      return 0.06;
    }
    if (cat === 'STUCK_CONCEPT_GAP') {
      if (style === 'socratic_question') return 0.28;
      if (style === 'concept_recall') return 0.24;
      if (style === 'condition_reminder') return 0.20;
      return 0.10;
    }

    if (input.triggerKind === 'pause_erase') {
      if (style === 'algebra_check') return 0.28;
      if (style === 'sanity_check') return 0.22;
      if (style === 'condition_reminder') return 0.18;
      return 0.12;
    }
    if (input.triggerKind === 'speed_drop') {
      if (style === 'step_decompose') return 0.26;
      if (style === 'socratic_question') return 0.22;
      return 0.12;
    }
    if (style === 'socratic_question') return 0.24;
    if (style === 'concept_recall') return 0.20;
    if (style === 'condition_reminder') return 0.18;
    return 0.12;
  }

  public updateFromFeedback(params: {
    timestamp: number;
    style: HintStyle;
    context: number[];
    signal: FeedbackSignal;
  }): number {
    const reward = this.signalToReward(params.signal);
    const arm = this.arms[params.style];
    const x = params.context.length === CONTEXT_DIM ? params.context : [1, 0.5, 0.5, 0, 0, 0, 1];

    for (let i = 0; i < CONTEXT_DIM; i++) {
      arm.b[i] += reward * x[i];
      const rowOffset = i * CONTEXT_DIM;
      for (let j = 0; j < CONTEXT_DIM; j++) {
        arm.aFlat[rowOffset + j] += x[i] * x[j];
      }
    }
    arm.pulls += 1;
    arm.rewardSum += reward;

    if (params.signal === 'dismiss_thinking') {
      this.thresholdOffset = clamp(this.thresholdOffset + 0.05, -0.15, 0.25);
    } else if (params.signal === 'dismiss_quick') {
      this.thresholdOffset = clamp(this.thresholdOffset + 0.03, -0.15, 0.25);
    } else if (params.signal === 'auto_resumed_writing') {
      this.thresholdOffset = clamp(this.thresholdOffset - 0.015, -0.15, 0.25);
    }

    this.feedbackLogs.push({
      timestamp: params.timestamp,
      style: params.style,
      signal: params.signal,
      reward,
      context: [...x],
    });
    if (this.feedbackLogs.length > 200) this.feedbackLogs.shift();

    return reward;
  }

  private signalToReward(signal: FeedbackSignal): number {
    switch (signal) {
      case 'auto_resumed_writing':
        return 1.0;
      case 'request_more':
        return 0.6;
      case 'dismiss_normal':
        return -0.1;
      case 'dismiss_quick':
        return -0.7;
      case 'dismiss_thinking':
        return -1.0;
    }
  }
}
