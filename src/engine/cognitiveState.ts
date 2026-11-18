import type { Diagnostics } from '../capture';
import type { Thresholds } from '../stalls';

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

/** 连续认知潜空间状态 s(t) = [M, R, A] */
export interface CognitiveState {
  momentumM: number;
  resistanceR: number;
  alignmentA: number;
  surpriseScore: number;
  pauseZScore: number;
  eraseStress: number;
  interventionScore: number;
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
 * 连续潜空间认知状态机（内环运动指纹 + 三维连续状态演化）。
 *
 * 【自研实现】设计思路：
 * 1. 内环（MotorFingerprint）：在每次自然停顿结束或擦除重写发生时用 EMA 更新个人基线，将原始停顿转化为个人 Z 分数；
 * 2. 状态转移：以微分平滑方式演化三维连续状态 s(t) = [M, R, A]；
 * 3. 语义注入槽：预留 setSemanticAlignment() 供后续接入 Google MathWriting Conformer 编码器。
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
    interventionScore: 0,
  };

  private mode: CompanionMode = 'socratic';
  private lastUpdateAt: number | null = null;
  private prevPauseMs = 0;
  private prevRewritten = false;
  private recentRewriteTimestamps: number[] = [];
  private lastEraseSampleAt: number | null = null;

  public resetSessionState(): void {
    this.state = {
      momentumM: 0.5,
      resistanceR: 0,
      alignmentA: 0,
      surpriseScore: 0,
      pauseZScore: 0,
      eraseStress: 0,
      interventionScore: 0,
    };
    this.lastUpdateAt = null;
    this.prevPauseMs = 0;
    this.prevRewritten = false;
    this.recentRewriteTimestamps = [];
    this.lastEraseSampleAt = null;
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
      this.fingerprint.speedRatioEma =
        0.95 * this.fingerprint.speedRatioEma + 0.05 * clamp(diag.speedRatio, 0.2, 2.0);
    }

    // 2. 个人分位数 Z 分数归一化与认知惊奇度
    const pauseRefMs = Math.max(diag.pauseMs, 150);
    const rawZPause =
      diag.pauseMs > 0
        ? (Math.log(pauseRefMs) - this.fingerprint.pauseLogMean) / this.fingerprint.pauseLogStd
        : -1.0;

    const personalWeight = clamp(this.fingerprint.sampleCount / 12, 0.2, 0.85);
    const stallRatio = diag.pauseMs / Math.max(thresholds.stallMs, 500);
    const surpriseScore =
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
    const eraseStress = clamp(excessErase * 0.32 + (diag.rewritten ? 0.45 : 0), 0, 1);

    // 3. 三维认知潜空间连续微分更新
    const activelyWriting = diag.rate >= 2 && diag.pauseMs < 450;
    const targetM = activelyWriting
      ? clamp(diag.speedRatio ?? 0.85, 0.15, 1.0)
      : Math.max(0, 1 - stallRatio * 0.65);
    const tauM = activelyWriting ? 1.8 : 4.5;
    this.state.momentumM = clamp(
      this.state.momentumM + ((targetM - this.state.momentumM) * dtSec) / tauM,
      0,
      1,
    );

    const speedStress =
      activelyWriting && diag.speedRatio !== null && diag.speedRatio < thresholds.speedRatio
        ? clamp((thresholds.speedRatio - diag.speedRatio) * 2.2, 0, 0.85)
        : 0;
    const pauseStress = sigmoid(1.35 * surpriseScore - 1.1) * 0.85;
    const targetR = clamp(Math.max(pauseStress, eraseStress, speedStress), 0, 1);
    const tauR = targetR > this.state.resistanceR ? 1.4 : 5.2;
    this.state.resistanceR = clamp(
      this.state.resistanceR + ((targetR - this.state.resistanceR) * dtSec) / tauR,
      0,
      1,
    );

    const modeBias = this.mode === 'coach' ? 0.14 : this.mode === 'zen' ? -0.18 : 0;
    const rawScore =
      1.9 * this.state.resistanceR +
      0.85 * (1 - this.state.momentumM) +
      1.1 * Math.max(0, -this.state.alignmentA) +
      modeBias -
      1.35;

    this.state.pauseZScore = Number(rawZPause.toFixed(2));
    this.state.eraseStress = Number(eraseStress.toFixed(2));
    this.state.surpriseScore = Number(surpriseScore.toFixed(2));
    this.state.interventionScore = Number(sigmoid(rawScore).toFixed(3));

    return this.getState();
  }
}
