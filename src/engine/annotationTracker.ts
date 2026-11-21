import type { Stroke } from '../capture';
import type { CognitiveState } from './cognitiveState';
import { computeBoundingBox, type BoundingBox } from './inkSyntaxEncoder';

export type AnnotationLabel =
  | 'FLUENT_DERIVATION'
  | 'DEEP_THINKING_HOLD'
  | 'CALC_MICRO_HESITATION'
  | 'SELF_CORRECTION'
  | 'STUCK_CONCEPT_GAP'
  | 'STUCK_CALCULATION'
  | 'DIRECTION_DRIFT_PITFALL'
  | 'HINT_EFFECTIVE'
  | 'PITFALL_TRIGGERED';

export const ANNOTATION_LABEL_ZH: Record<AnnotationLabel, string> = {
  FLUENT_DERIVATION: '顺畅推导',
  DEEP_THINKING_HOLD: '深度心算',
  CALC_MICRO_HESITATION: '算式微卡犹豫',
  SELF_CORRECTION: '自我涂改纠错',
  STUCK_CONCEPT_GAP: '思路/定理卡壳',
  STUCK_CALCULATION: '算式/化简受阻',
  DIRECTION_DRIFT_PITFALL: '路径偏离/歧路',
  HINT_EFFECTIVE: '提示后恢复推导',
  PITFALL_TRIGGERED: '触碰易错歧路',
};

export interface AnnotationRecord {
  id: string;
  timestamp: number;
  label: AnnotationLabel;
  source:
    | 'user_tap_button'
    | 'eraser_event'
    | 'undo_event'
    | 'implicit_recovery'
    | 'auto_fluent'
    | 'syntax_pitfall';
  durationMs?: number;
  strokeId?: number;
  bbox?: BoundingBox;
  note?: string;
}

export interface StateTraceSample {
  t: number;
  M: number;
  R: number;
  A: number;
  S: number;
  P: number;
  matchedStepIndex: number;
}

const MAX_TRACE_POINTS = 2400;
const TRACE_MIN_INTERVAL_MS = 500;
const FLUENT_SUSTAIN_MS = 6000;
const FLUENT_COOLDOWN_MS = 12000;

/**
 * 零负担伴随式微标注协议与三维状态轨迹采集器。
 *
 * 【自研实现】设计思路：
 * 1. 自动捕获隐式行为标签（橡皮擦/撤销 → SELF_CORRECTION；提示后 5 秒恢复书写 → HINT_EFFECTIVE；
 *    连续高动量低阻力书写 → FLUENT_DERIVATION），无需学生事后手工逐帧打标；
 * 2. 兼容前端伴随胶囊的少量显式标签（DEEP_THINKING_HOLD / STUCK_CONCEPT_GAP / STUCK_CALCULATION），
 *    用于精确区分“深度心算勿扰”与“需要启发”两类核心训练样本；
 * 3. 以防内存膨胀的自适应降采样频率记录 [M, R, A, S, P] 轨迹，可直接导出用于后续 ML 训练评估。
 */
export class AnnotationTracker {
  private annotations: AnnotationRecord[] = [];
  private stateTrace: StateTraceSample[] = [];
  private nextSeq = 1;
  private lastTraceAt = 0;
  private lastTraceState: CognitiveState | null = null;
  private fluentStartedAt: number | null = null;
  private lastFluentAnnotatedAt = 0;

  public reset(): void {
    this.annotations = [];
    this.stateTrace = [];
    this.nextSeq = 1;
    this.lastTraceAt = 0;
    this.lastTraceState = null;
    this.fluentStartedAt = null;
    this.lastFluentAnnotatedAt = 0;
  }

  public addAnnotation(input: Omit<AnnotationRecord, 'id'>): AnnotationRecord {
    const rec: AnnotationRecord = {
      id: `ann_${String(this.nextSeq++).padStart(3, '0')}`,
      ...input,
    };
    this.annotations.push(rec);
    return rec;
  }

  public recordEraseOrUndo(
    stroke: Stroke,
    kind: 'eraser_event' | 'undo_event',
    nowMs = Date.now(),
  ): AnnotationRecord {
    this.fluentStartedAt = null;
    const t = stroke.points[stroke.points.length - 1]?.t ?? nowMs;
    const firstT = stroke.points[0]?.t ?? t;
    return this.addAnnotation({
      timestamp: t,
      label: 'SELF_CORRECTION',
      source: kind,
      durationMs: Math.max(0, t - firstT),
      strokeId: stroke.id,
      bbox: computeBoundingBox(stroke.points),
    });
  }

  public recordDiagnosticsSample(
    nowMs: number,
    state: CognitiveState,
    matchedStepIndex: number,
    activelyWriting: boolean,
  ): void {
    // 1. 自监督顺畅推导区间采集
    if (activelyWriting && state.momentumM >= 0.68 && state.resistanceR <= 0.25) {
      if (this.fluentStartedAt === null) {
        this.fluentStartedAt = nowMs;
      }
      if (
        nowMs - this.fluentStartedAt >= FLUENT_SUSTAIN_MS &&
        nowMs - this.lastFluentAnnotatedAt >= FLUENT_COOLDOWN_MS
      ) {
        this.lastFluentAnnotatedAt = nowMs;
        this.fluentStartedAt = nowMs;
        this.addAnnotation({
          timestamp: nowMs,
          label: 'FLUENT_DERIVATION',
          source: 'auto_fluent',
          durationMs: FLUENT_SUSTAIN_MS,
        });
      }
    } else if (state.resistanceR > 0.38) {
      this.fluentStartedAt = null;
    }

    // 2. 自适应节流采样三维状态演化轨迹
    const deltaBig =
      this.lastTraceState !== null &&
      (Math.abs(state.momentumM - this.lastTraceState.momentumM) >= 0.08 ||
        Math.abs(state.resistanceR - this.lastTraceState.resistanceR) >= 0.08 ||
        Math.abs(state.alignmentA - this.lastTraceState.alignmentA) >= 0.1 ||
        Math.abs(state.interventionScore - this.lastTraceState.interventionScore) >= 0.08);

    if (nowMs - this.lastTraceAt < TRACE_MIN_INTERVAL_MS && !deltaBig) {
      return;
    }

    this.lastTraceAt = nowMs;
    this.lastTraceState = { ...state };
    if (this.stateTrace.length >= MAX_TRACE_POINTS) {
      this.stateTrace.shift();
    }
    this.stateTrace.push({
      t: nowMs,
      M: Number(state.momentumM.toFixed(3)),
      R: Number(state.resistanceR.toFixed(3)),
      A: Number(state.alignmentA.toFixed(3)),
      S: Number(state.surpriseScore.toFixed(3)),
      P: Number(state.interventionScore.toFixed(3)),
      matchedStepIndex,
    });
  }

  public getAnnotations(): AnnotationRecord[] {
    return [...this.annotations];
  }

  public getStateTrace(): StateTraceSample[] {
    return [...this.stateTrace];
  }

  public getCountsByLabel(): Record<AnnotationLabel, number> {
    const counts: Record<AnnotationLabel, number> = {
      FLUENT_DERIVATION: 0,
      DEEP_THINKING_HOLD: 0,
      CALC_MICRO_HESITATION: 0,
      SELF_CORRECTION: 0,
      STUCK_CONCEPT_GAP: 0,
      STUCK_CALCULATION: 0,
      DIRECTION_DRIFT_PITFALL: 0,
      HINT_EFFECTIVE: 0,
      PITFALL_TRIGGERED: 0,
    };
    for (const item of this.annotations) {
      counts[item.label] += 1;
    }
    return counts;
  }
}
