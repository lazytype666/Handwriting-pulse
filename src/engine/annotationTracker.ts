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
  anchorGap?: number;
  stepProgress?: number;
}

const MAX_TRACE_POINTS = 2400;
const TRACE_MIN_INTERVAL_MS = 500;
const FLUENT_SUSTAIN_MS = 6000;
const FLUENT_COOLDOWN_MS = 12000;

/**
 * 零负担伴随式微标注协议与三维状态轨迹采集器。
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
      anchorGap: Number(state.anchorGap.toFixed(3)),
      stepProgress: Number(state.stepProgress.toFixed(3)),
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
