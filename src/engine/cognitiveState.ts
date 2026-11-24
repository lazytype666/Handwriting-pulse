import type { Diagnostics } from '../capture';
import type { Thresholds } from '../stalls';
import type { CognitiveCategory, NeuralCognitiveOutput } from './neuralRuntime';

export type { CognitiveCategory, NeuralCognitiveOutput };

/** 个性化伴随模式：给不同做题习惯的用户留出自主选择空间 */
export type CompanionMode = 'zen' | 'socratic' | 'coach';

export const COMPANION_MODE_LABELS: Record<CompanionMode, string> = {
  zen: '探索',
  socratic: '苏格拉底',
  coach: '教练',
};

/** 个人运动指纹：内环自适应学习用户的停顿分布、书写速度比常态与涂改频率 */
export interface MotorFingerprint {
  sampleCount: number;
  pauseLogMean: number;
  pauseLogStd: number;
  speedRatioEma: number;
  eraseBaselineCount: number;
}

/** 连续认知潜空间状态 s(t) = [M, R, A]、解题图谱对照反馈指标及六类细粒度认知状态 */
export interface CognitiveState {
  momentumM: number;
  resistanceR: number;
  alignmentA: number;
  surpriseScore: number;
  pauseZScore: number;
  eraseStress: number;
  speedInstability: number;
  prePauseMomentum: number;
  interventionScore: number;
  structuralBoundaryFactor: number;
  matchedStepIndex: number;
  recognizedLatex: string;
  /** 图谱期望步骤缺口 AnchorGap ∈ [0, 1]（期待锚点权重与当前观测后验之差） */
  anchorGap: number;
  /** 步骤停滞度 StepStagnation ∈ [0, 1]（同一步骤长期无有效推进） */
  stepStagnation: number;
  /** 步骤跃迁正向脉冲 StepAdvancePulse ∈ [0, 1]（刚跨步后指数衰减，防顿悟心算误报） */
  stepAdvancePulse: number;
  /** 解题步骤归一化进度 StepProgress ∈ [0, 1] */
  stepProgress: number;
  /** 连续易错歧路概率 PitfallProb ∈ [0, 1] */
  pitfallProb: number;
  cognitiveCategory: CognitiveCategory;
  neuralMode: 'neural_ts' | 'heuristic';
  neuralClassProbs?: number[];
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

const DEFAULT_FINGERPRINT: MotorFingerprint = {
  sampleCount: 0,
  pauseLogMean: Math.log(1200),
  pauseLogStd: 0.75,
  speedRatioEma: 1.0,
  eraseBaselineCount: 0.35,
};

/**
 * 连续潜空间认知状态机（内环运动指纹 + 动力学/图谱双闭环状态演化）。
 *
 * 【自研实现】设计思路：
 * 1. 内环（MotorFingerprint）：在每次自然停顿结束或擦除重写发生时用 EMA 更新个人基线，将原始停顿转化为个人 Z 分数；
 * 2. 图谱反馈注入（图四闭环）：接入 `anchorGap`、`stepStagnation`、`stepAdvancePulse`、`pitfallProb`，
 *    在“跨步心算保护”、“长期踏步识别”与“思路卡壳 vs 计算受阻消歧”上直接约束状态转移与干预概率；
 * 3. 状态转移：以微分平滑方式演化三维连续状态 s(t) = [M, R, A] 并融合 14 维 `CognitiveStateNet` 神经输出。
 */
export class CognitiveStateEngine {
  private fingerprint: MotorFingerprint = { ...DEFAULT_FINGERPRINT };
  private state: CognitiveState = {
    momentumM: 0.5,
    resistanceR: 0,
    alignmentA: 0,
    surpriseScore: 0,
    pauseZScore: 0,
    eraseStress: 0,
    speedInstability: 0,
    prePauseMomentum: 0.5,
    interventionScore: 0,
    structuralBoundaryFactor: 1.0,
    matchedStepIndex: -1,
    recognizedLatex: '',
    anchorGap: 0,
    stepStagnation: 0,
    stepAdvancePulse: 0,
    stepProgress: 0,
    pitfallProb: 0,
    cognitiveCategory: 'FLUENT_DERIVATION',
    neuralMode: 'heuristic',
    neuralClassProbs: undefined,
  };

  private mode: CompanionMode = 'socratic';
  private lastUpdateAt: number | null = null;
  private prevPauseMs = 0;
  private prevRewritten = false;
  private wasActivelyWriting = false;
  private speedVarEma = 0;
  private prePauseMomentum = 0.5;
  private recentRewriteTimestamps: number[] = [];
  private lastEraseSampleAt: number | null = null;
  private lastNeuralOutput: NeuralCognitiveOutput | null = null;
  private lastNeuralAppliedAt = 0;
  private lastStepAdvancedAt: number | null = null;

  public resetSessionState(): void {
    this.state = {
      momentumM: 0.5,
      resistanceR: 0,
      alignmentA: 0,
      surpriseScore: 0,
      pauseZScore: 0,
      eraseStress: 0,
      speedInstability: 0,
      prePauseMomentum: 0.5,
      interventionScore: 0,
      structuralBoundaryFactor: 1.0,
      matchedStepIndex: -1,
      recognizedLatex: '',
      anchorGap: 0,
      stepStagnation: 0,
      stepAdvancePulse: 0,
      stepProgress: 0,
      pitfallProb: 0,
      cognitiveCategory: 'FLUENT_DERIVATION',
      neuralMode: 'heuristic',
      neuralClassProbs: undefined,
    };
    this.lastUpdateAt = null;
    this.prevPauseMs = 0;
    this.prevRewritten = false;
    this.wasActivelyWriting = false;
    this.speedVarEma = 0;
    this.prePauseMomentum = 0.5;
    this.recentRewriteTimestamps = [];
    this.lastEraseSampleAt = null;
    this.lastNeuralOutput = null;
    this.lastNeuralAppliedAt = 0;
    this.lastStepAdvancedAt = null;
  }

  public resetFingerprint(): void {
    this.fingerprint = { ...DEFAULT_FINGERPRINT };
  }

  public hydrateFingerprint(saved: Partial<MotorFingerprint> | null): void {
    if (!saved) return;
    this.fingerprint = {
      sampleCount: saved.sampleCount ?? 0,
      pauseLogMean: saved.pauseLogMean ?? DEFAULT_FINGERPRINT.pauseLogMean,
      pauseLogStd: clamp(saved.pauseLogStd ?? DEFAULT_FINGERPRINT.pauseLogStd, 0.35, 1.4),
      speedRatioEma: clamp(saved.speedRatioEma ?? DEFAULT_FINGERPRINT.speedRatioEma, 0.3, 2.0),
      eraseBaselineCount: clamp(
        saved.eraseBaselineCount ?? DEFAULT_FINGERPRINT.eraseBaselineCount,
        0.1,
        5,
      ),
    };
  }

  public setCompanionMode(mode: CompanionMode): void {
    this.mode = mode;
  }

  public getFingerprint(): MotorFingerprint {
    return { ...this.fingerprint };
  }

  public getState(): CognitiveState {
    return { ...this.state };
  }

  public setSemanticAlignment(alignmentA: number): void {
    this.state.alignmentA = clamp(alignmentA, -1, 1);
  }

  public setSyntaxObservation(
    obs: {
      alignmentA: number;
      structuralBoundaryFactor?: number;
      matchedStepIndex?: number;
      recognizedLatex?: string;
      anchorGap?: number;
      stepProgress?: number;
      pitfallProb?: number;
      stepStagnation?: number;
      stepAdvancePulse?: number;
    },
    nowMs = Date.now(),
  ): void {
    const prevStep = this.state.matchedStepIndex;
    this.state.alignmentA = clamp(obs.alignmentA, -1, 1);
    if (obs.structuralBoundaryFactor !== undefined) {
      this.state.structuralBoundaryFactor = clamp(obs.structuralBoundaryFactor, 0.7, 1.8);
    }
    if (obs.matchedStepIndex !== undefined) {
      if (obs.matchedStepIndex > prevStep && obs.matchedStepIndex >= 0) {
        this.state.stepAdvancePulse = 1.0;
        this.state.stepStagnation = Math.min(this.state.stepStagnation, 0.12);
        this.lastStepAdvancedAt = nowMs;
      }
      this.state.matchedStepIndex = obs.matchedStepIndex;
    }
    if (obs.recognizedLatex !== undefined) {
      this.state.recognizedLatex = obs.recognizedLatex;
    }
    if (obs.anchorGap !== undefined) {
      this.state.anchorGap = Number(clamp(obs.anchorGap, 0, 1).toFixed(3));
    }
    if (obs.stepProgress !== undefined) {
      const clampedProg = clamp(obs.stepProgress, 0, 1);
      if (clampedProg - this.state.stepProgress >= 0.15) {
        this.state.stepAdvancePulse = Math.max(this.state.stepAdvancePulse, 0.9);
        this.lastStepAdvancedAt = nowMs;
      }
      this.state.stepProgress = Number(clampedProg.toFixed(3));
    }
    if (obs.pitfallProb !== undefined) {
      this.state.pitfallProb = Number(clamp(obs.pitfallProb, 0, 1).toFixed(3));
    }
    if (obs.stepStagnation !== undefined) {
      this.state.stepStagnation = Number(clamp(obs.stepStagnation, 0, 1).toFixed(3));
    }
    if (obs.stepAdvancePulse !== undefined) {
      this.state.stepAdvancePulse = Number(clamp(obs.stepAdvancePulse, 0, 1).toFixed(3));
    }
    this.state.interventionScore = this.computeInterventionProbability(this.state.cognitiveCategory);
  }

  /**
   * 融合端侧 `CognitiveStateNet` (14 维图谱融合版) 神经分类输出，将 6 类认知 logits、惊奇度与潜空间 [M, R, A] 并入状态机。
   */
  public applyNeuralCognitiveOutput(output: NeuralCognitiveOutput, nowMs = Date.now()): CognitiveState {
    this.lastNeuralOutput = output;
    this.lastNeuralAppliedAt = nowMs;

    this.state.neuralMode = 'neural_ts';
    this.state.cognitiveCategory = output.category;
    this.state.neuralClassProbs = [...output.classProbs];
    this.state.momentumM = Number(clamp(0.42 * this.state.momentumM + 0.58 * output.momentumM, 0, 1).toFixed(3));
    this.state.resistanceR = Number(clamp(0.42 * this.state.resistanceR + 0.58 * output.resistanceR, 0, 1).toFixed(3));

    // 若已命中明确的歧路负对齐 (-0.45 以下) 或连续易错高概率，保留其方向锚点并与神经流预测融合
    if (this.state.alignmentA <= -0.45 || this.state.pitfallProb >= 0.68) {
      this.state.alignmentA = Number(clamp(Math.min(this.state.alignmentA, output.alignmentA), -1, 1).toFixed(2));
      this.state.cognitiveCategory = 'DIRECTION_DRIFT_PITFALL';
    } else if (Math.abs(this.state.alignmentA) > 0.05) {
      this.state.alignmentA = Number(clamp(0.55 * this.state.alignmentA + 0.45 * output.alignmentA, -1, 1).toFixed(2));
    } else {
      this.state.alignmentA = Number(clamp(output.alignmentA, -1, 1).toFixed(2));
    }

    this.state.surpriseScore = Number(
      clamp(0.35 * this.state.surpriseScore + 0.65 * output.surpriseScore, 0, 3).toFixed(2),
    );

    this.state.interventionScore = this.computeInterventionProbability(this.state.cognitiveCategory);
    return this.getState();
  }

  private computeInterventionProbability(category: CognitiveCategory): number {
    const modeBias = this.mode === 'coach' ? 0.14 : this.mode === 'zen' ? -0.18 : 0;
    const rawScore =
      1.85 * this.state.resistanceR +
      0.75 * (1 - this.state.momentumM) +
      1.1 * Math.max(0, -this.state.alignmentA) +
      0.65 * this.state.stepStagnation * this.state.anchorGap +
      0.55 * this.state.pitfallProb -
      0.6 * this.state.stepAdvancePulse +
      modeBias -
      1.38;
    let pInt = sigmoid(rawScore);
    if (category === 'DEEP_THINKING_HOLD') {
      pInt *= 0.48;
    } else if (category === 'FLUENT_DERIVATION') {
      pInt *= 0.65;
    } else if (
      category === 'DIRECTION_DRIFT_PITFALL' &&
      (this.state.alignmentA <= -0.45 || this.state.pitfallProb >= 0.68)
    ) {
      pInt = Math.max(pInt, 0.74);
    }
    if (this.state.stepAdvancePulse >= 0.5 && category !== 'DIRECTION_DRIFT_PITFALL') {
      pInt *= 0.62;
    }
    return Number(clamp(pInt, 0, 1).toFixed(3));
  }

  public updateFromDiagnostics(
    nowMs: number,
    diag: Diagnostics,
    thresholds: Thresholds,
  ): CognitiveState {
    const dtSec =
      this.lastUpdateAt !== null
        ? clamp((nowMs - this.lastUpdateAt) / 1000, 0.05, 1.0)
        : 0.125;
    this.lastUpdateAt = nowMs;

    // 0. 衰减刚推进步骤的正向脉冲 (τ = 4.5s)，并在具备图谱进度时平滑追踪 stepStagnation
    this.state.stepAdvancePulse = Number(
      clamp(this.state.stepAdvancePulse * Math.exp(-dtSec / 4.5), 0, 1).toFixed(3),
    );
    if (this.lastStepAdvancedAt === null && (this.state.matchedStepIndex >= 0 || this.state.anchorGap > 0.1)) {
      this.lastStepAdvancedAt = nowMs;
    }
    if (this.lastStepAdvancedAt !== null && this.state.stepProgress < 0.98) {
      const silentStepMs = Math.max(0, nowMs - this.lastStepAdvancedAt);
      const stagnationHorizon = Math.max(10000, thresholds.stallMs * 4.0);
      const rawStag = clamp((silentStepMs - thresholds.stallMs * 1.5) / stagnationHorizon, 0, 1);
      const modulated = rawStag * (0.4 + 0.6 * Math.max(this.state.anchorGap, 0.25));
      this.state.stepStagnation = Number(
        clamp(Math.max(this.state.stepStagnation * Math.exp(-dtSec / 14.0), modulated), 0, 1).toFixed(3),
      );
    }

    // 1. 内环更新：检测到一次自然停顿结束（由 >=700ms 突然回落到新触点 <350ms）时更新个人停顿分布
    if (this.prevPauseMs >= 700 && this.prevPauseMs < 25000 && diag.pauseMs < 350) {
      const logDur = Math.log(Math.max(this.prevPauseMs, 150));
      const alpha = this.fingerprint.sampleCount < 10 ? 0.25 : 0.08;
      const diff = logDur - this.fingerprint.pauseLogMean;
      this.fingerprint.pauseLogMean += alpha * diff;
      const nextVar =
        (1 - alpha) * Math.pow(this.fingerprint.pauseLogStd, 2) + alpha * diff * diff;
      this.fingerprint.pauseLogStd = clamp(Math.sqrt(nextVar), 0.35, 1.4);
      this.fingerprint.sampleCount += 1;
    }
    this.prevPauseMs = diag.pauseMs;

    if (diag.rewritten && !this.prevRewritten) {
      this.recentRewriteTimestamps.push(nowMs);
    }
    this.prevRewritten = diag.rewritten;
    while (
      this.recentRewriteTimestamps.length > 0 &&
      nowMs - this.recentRewriteTimestamps[0] > 15000
    ) {
      this.recentRewriteTimestamps.shift();
    }

    if (this.lastEraseSampleAt === null || nowMs - this.lastEraseSampleAt >= 3000) {
      this.lastEraseSampleAt = nowMs;
      this.fingerprint.eraseBaselineCount =
        0.92 * this.fingerprint.eraseBaselineCount +
        0.08 * this.recentRewriteTimestamps.length;
    }

    if (diag.speedRatio !== null && diag.rate > 0) {
      const clampedSpeed = clamp(diag.speedRatio, 0.2, 2.0);
      const speedDev = Math.abs(clampedSpeed - this.fingerprint.speedRatioEma);
      this.speedVarEma = clamp(0.82 * this.speedVarEma + 0.18 * speedDev, 0, 1);
      this.fingerprint.speedRatioEma = 0.95 * this.fingerprint.speedRatioEma + 0.05 * clampedSpeed;
    }

    const activelyWriting = diag.rate >= 2 && diag.pauseMs < 450;
    if (this.wasActivelyWriting && !activelyWriting) {
      // 记录进入停顿前那一瞬间的书写流畅惯性，区分“顺畅落笔后心算停顿”与“破碎犹豫后停笔”
      this.prePauseMomentum = this.state.momentumM;
    } else if (activelyWriting) {
      this.prePauseMomentum = clamp(0.8 * this.prePauseMomentum + 0.2 * this.state.momentumM, 0, 1);
    }
    this.wasActivelyWriting = activelyWriting;

    // 2. 个人分位数 Z 分数归一化 + 结构边界惊奇度 + 深度心算与步骤跨跃保护带
    const pauseRefMs = Math.max(diag.pauseMs, 150);
    const rawZPause =
      diag.pauseMs > 0
        ? (Math.log(pauseRefMs) - this.fingerprint.pauseLogMean) / this.fingerprint.pauseLogStd
        : -1.0;

    const personalWeight = clamp(this.fingerprint.sampleCount / 12, 0.2, 0.85);
    const stallRatio = diag.pauseMs / Math.max(thresholds.stallMs, 500);
    const baseSurprise =
      diag.pauseMs > 0
        ? Math.max(
            0,
            personalWeight * Math.max(0, rawZPause) +
              (1 - personalWeight) * 1.6 * Math.max(0, stallRatio - 0.5),
          )
        : 0;

    const excessErase = Math.max(
      0,
      this.recentRewriteTimestamps.length - Math.max(0.5, this.fingerprint.eraseBaselineCount),
    );
    const eraseStress = clamp(excessErase * 0.34 + (diag.rewritten ? 0.48 : 0), 0, 1);

    // 深度心算阻尼：在数字运算中间停顿、前序动量充沛且无涂改时衰减惊奇度，减少心算误报
    const inHealthyMentalCalc =
      eraseStress < 0.2 &&
      this.state.structuralBoundaryFactor < 1.0 &&
      this.prePauseMomentum >= 0.52 &&
      rawZPause < 2.05;
    const mentalCalcShield = inHealthyMentalCalc ? 0.68 : 1.0;
    // 图谱跨步保护盾：刚跨过一个步骤节点后的 4~6 秒内认为正在酝酿下一步心算，避免误判卡壳
    const stepJumpShield = this.state.stepAdvancePulse > 0.35 ? 0.55 : 1.0;
    const hesitationBoost = this.prePauseMomentum < 0.45 ? 0.25 * (1 - this.prePauseMomentum) : 0;
    const surpriseScore =
      baseSurprise * this.state.structuralBoundaryFactor * mentalCalcShield * stepJumpShield +
      hesitationBoost;

    // 3. 三维认知潜空间连续微分更新（引入图谱步骤长期停滞与锚点缺口压力 stagnationStress）
    const adaptivePauseDecay = clamp(Math.max(0, rawZPause) / 2.5, 0, 1);
    const targetM = activelyWriting
      ? clamp(diag.speedRatio ?? 0.85, 0.15, 1.0)
      : Math.max(0, 1 - (0.65 * adaptivePauseDecay + 0.35 * clamp(stallRatio * 0.65, 0, 1)));
    const tauM = activelyWriting ? 1.8 : 4.0;
    this.state.momentumM = clamp(
      this.state.momentumM + ((targetM - this.state.momentumM) * dtSec) / tauM,
      0,
      1,
    );

    const speedStress =
      activelyWriting && diag.speedRatio !== null && diag.speedRatio < thresholds.speedRatio
        ? clamp((thresholds.speedRatio - diag.speedRatio) * 2.2 + this.speedVarEma * 0.45, 0, 0.9)
        : 0;
    const pauseStress = sigmoid(1.45 * surpriseScore - 1.25) * 0.88;
    const instabilityStress = clamp(this.speedVarEma * 1.15, 0, 0.65);
    const stagnationStress = clamp(this.state.stepStagnation * (0.45 + 0.55 * this.state.anchorGap), 0, 0.88);
    const targetR = clamp(
      Math.max(pauseStress, eraseStress, speedStress, instabilityStress * 0.8, stagnationStress),
      0,
      1,
    );
    const tauR = targetR > this.state.resistanceR ? 1.35 : 5.0;
    this.state.resistanceR = clamp(
      this.state.resistanceR + ((targetR - this.state.resistanceR) * dtSec) / tauR,
      0,
      1,
    );

    this.state.pauseZScore = Number(rawZPause.toFixed(2));
    this.state.eraseStress = Number(eraseStress.toFixed(2));
    this.state.speedInstability = Number(this.speedVarEma.toFixed(2));
    this.state.prePauseMomentum = Number(this.prePauseMomentum.toFixed(2));
    this.state.surpriseScore = Number(surpriseScore.toFixed(2));

    let nextCategory: CognitiveCategory = this.state.cognitiveCategory;
    if (!this.lastNeuralOutput) {
      this.state.neuralMode = 'heuristic';
      if (this.state.alignmentA <= -0.45 || this.state.pitfallProb >= 0.68) {
        nextCategory = 'DIRECTION_DRIFT_PITFALL';
      } else if (activelyWriting && this.state.stepStagnation < 0.72) {
        nextCategory = 'FLUENT_DERIVATION';
      } else if (
        (this.state.stepAdvancePulse >= 0.45 && eraseStress < 0.35) ||
        inHealthyMentalCalc ||
        (diag.pauseMs >= 700 && eraseStress < 0.2 && this.prePauseMomentum >= 0.52 && rawZPause < 2.05)
      ) {
        nextCategory = 'DEEP_THINKING_HOLD';
      } else if (this.state.stepStagnation >= 0.55 && this.state.anchorGap >= 0.55) {
        nextCategory = 'STUCK_CONCEPT_GAP';
      } else if (
        eraseStress >= 0.45 &&
        (this.speedVarEma >= 0.14 ||
          this.state.resistanceR >= 0.45 ||
          (this.state.anchorGap < 0.35 && this.state.stepStagnation < 0.4))
      ) {
        nextCategory = 'STUCK_CALCULATION';
      } else if (diag.pauseMs >= 1600 && rawZPause >= 1.8 && this.state.structuralBoundaryFactor >= 1.15) {
        nextCategory = 'STUCK_CONCEPT_GAP';
      } else if (diag.pauseMs >= 1100 && (rawZPause >= 1.25 || eraseStress >= 0.25)) {
        nextCategory = 'CALC_MICRO_HESITATION';
      } else {
        nextCategory = 'FLUENT_DERIVATION';
      }
    } else if (this.state.alignmentA <= -0.45 || this.state.pitfallProb >= 0.68) {
      nextCategory = 'DIRECTION_DRIFT_PITFALL';
    } else if (
      activelyWriting &&
      this.state.stepStagnation < 0.72 &&
      nowMs - this.lastNeuralAppliedAt >= 900 &&
      this.state.momentumM >= 0.55
    ) {
      nextCategory = 'FLUENT_DERIVATION';
    } else if (!activelyWriting && diag.pauseMs >= 1100 && nowMs - this.lastNeuralAppliedAt >= 950) {
      if (this.state.stepAdvancePulse >= 0.45 && eraseStress < 0.35) {
        nextCategory = 'DEEP_THINKING_HOLD';
      } else if (
        this.lastNeuralOutput.category === 'DEEP_THINKING_HOLD' &&
        eraseStress < 0.25 &&
        rawZPause < 2.25
      ) {
        nextCategory = 'DEEP_THINKING_HOLD';
      } else if (inHealthyMentalCalc && rawZPause < 2.0) {
        nextCategory = 'DEEP_THINKING_HOLD';
      } else if (this.state.stepStagnation >= 0.55 && this.state.anchorGap >= 0.55) {
        nextCategory = 'STUCK_CONCEPT_GAP';
      } else if (eraseStress >= 0.45) {
        nextCategory = 'STUCK_CALCULATION';
      } else if (rawZPause >= 1.8 && this.state.structuralBoundaryFactor >= 1.15) {
        nextCategory = 'STUCK_CONCEPT_GAP';
      } else if (rawZPause >= 1.25) {
        nextCategory = 'CALC_MICRO_HESITATION';
      }
    }
    this.state.cognitiveCategory = nextCategory;
    this.state.interventionScore = this.computeInterventionProbability(nextCategory);

    return this.getState();
  }
}
