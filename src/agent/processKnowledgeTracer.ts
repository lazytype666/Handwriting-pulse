import type { HintStyle } from '../engine/personalizedPolicy';
import {
  KNOWLEDGE_TAG_LABELS,
  type KnowledgeConceptTag,
  type SessionLearningTelemetry,
  type WritingHabitMetrics,
} from './sessionAggregator';

export interface ConceptMasteryState {
  tag: KnowledgeConceptTag;
  label: string;
  masteryProb: number;
  attempts: number;
  pitfallHits: number;
  lastPracticedAt: number;
  streak: number;
  emaAlignment: number;
}

export type WritingDominantWeakness =
  | 'concept_gap'
  | 'calculation_carelessness'
  | 'boundary_oversight'
  | 'pace_instability'
  | 'balanced';

export interface WritingHabitProfile {
  sessionsCount: number;
  avgMomentum: number;
  avgEraseRatio: number;
  avgPauseZScore: number;
  structuralPauseRatio: number;
  deepThinkingRatio: number;
  pitfallFrequency: number;
  dominantWeakness: WritingDominantWeakness;
}

export interface SessionDigestSummary {
  sessionId: string;
  capturedAt: number;
  problemKey: string;
  knowledgeTags: KnowledgeConceptTag[];
  finalStepProgress: number;
  maxPitfallProb: number;
  erasePerStrokeRatio: number;
  durationMs: number;
}

export interface LearnerCognitiveTwin {
  schemaVersion: 'twin-v1';
  userId: string;
  updatedAt: number;
  totalSessions: number;
  totalProblemsSolved: number;
  conceptMastery: Record<string, ConceptMasteryState>;
  habitProfile: WritingHabitProfile;
  preferredHintStyle: HintStyle | null;
  recentSessions: SessionDigestSummary[];
}

export interface ProcessBktParams {
  pInit: number;
  pTransit: number;
  pSlip: number;
  pGuess: number;
}

const DEFAULT_BKT_PARAMS: ProcessBktParams = {
  pInit: 0.35,
  pTransit: 0.18,
  pSlip: 0.12,
  pGuess: 0.18,
};

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function classifyDominantWeakness(p: WritingHabitProfile): WritingDominantWeakness {
  if (p.sessionsCount === 0) return 'balanced';
  if (p.pitfallFrequency >= 0.42 && p.structuralPauseRatio >= 0.5) {
    return 'boundary_oversight';
  }
  if (p.avgEraseRatio >= 0.36) {
    return 'calculation_carelessness';
  }
  if (p.structuralPauseRatio >= 0.62 && p.avgPauseZScore >= 0.85) {
    return 'concept_gap';
  }
  if (p.avgMomentum < 0.42 && p.deepThinkingRatio < 0.25) {
    return 'pace_instability';
  }
  return 'balanced';
}

export function createInitialLearnerTwin(userId = 'default-learner'): LearnerCognitiveTwin {
  return {
    schemaVersion: 'twin-v1',
    userId,
    updatedAt: Date.now(),
    totalSessions: 0,
    totalProblemsSolved: 0,
    conceptMastery: {},
    habitProfile: {
      sessionsCount: 0,
      avgMomentum: 0.58,
      avgEraseRatio: 0.12,
      avgPauseZScore: 0,
      structuralPauseRatio: 0.4,
      deepThinkingRatio: 0.15,
      pitfallFrequency: 0,
      dominantWeakness: 'balanced',
    },
    preferredHintStyle: null,
    recentSessions: [],
  };
}

/**
 * 端侧过程性知识追踪器（Process-Aware Bayesian Knowledge Tracing + 书写习惯数字孪生）。
 */
export class ProcessKnowledgeTracer {
  private twin: LearnerCognitiveTwin;

  constructor(initialTwin?: LearnerCognitiveTwin | null) {
    this.twin = initialTwin
      ? {
          ...initialTwin,
          conceptMastery: { ...initialTwin.conceptMastery },
          habitProfile: { ...initialTwin.habitProfile },
          recentSessions: [...(initialTwin.recentSessions ?? [])],
        }
      : createInitialLearnerTwin();
  }

  public getTwin(): LearnerCognitiveTwin {
    return {
      ...this.twin,
      conceptMastery: { ...this.twin.conceptMastery },
      habitProfile: { ...this.twin.habitProfile },
      recentSessions: [...this.twin.recentSessions],
    };
  }

  public hydrateTwin(saved: LearnerCognitiveTwin | null | undefined): void {
    if (!saved || saved.schemaVersion !== 'twin-v1') return;
    this.twin = {
      ...saved,
      conceptMastery: { ...(saved.conceptMastery ?? {}) },
      habitProfile: {
        ...createInitialLearnerTwin(saved.userId).habitProfile,
        ...(saved.habitProfile ?? {}),
      },
      recentSessions: [...(saved.recentSessions ?? [])].slice(-20),
    };
  }

  public computeSoftObservationScore(telemetry: SessionLearningTelemetry): number {
    const progress = clamp(telemetry.finalStepProgress, 0, 1);
    const alignNorm = clamp((telemetry.meanAlignmentA + 1) / 2, 0, 1);
    const stuckFrac = clamp(
      telemetry.habits.stuckConceptGapRatio + telemetry.habits.stuckCalculationRatio * 0.7,
      0,
      1,
    );
    const pitfallPenalty =
      telemetry.hitPitfalls.length > 0
        ? clamp(0.28 + 0.25 * telemetry.maxPitfallProb, 0.28, 0.55)
        : Math.max(0, telemetry.maxPitfallProb - 0.5) * 0.35;

    return Number(
      clamp(
        0.46 * progress + 0.3 * alignNorm + 0.24 * (1 - stuckFrac) - pitfallPenalty,
        0.05,
        0.96,
      ).toFixed(3),
    );
  }

  private bktUpdateStep(prevMastery: number, softCorrectProb: number, p = DEFAULT_BKT_PARAMS): number {
    const lPrev = clamp(prevMastery, 0.02, 0.98);
    const pObs1 = (lPrev * (1 - p.pSlip)) / Math.max(1e-6, lPrev * (1 - p.pSlip) + (1 - lPrev) * p.pGuess);
    const pObs0 = (lPrev * p.pSlip) / Math.max(1e-6, lPrev * p.pSlip + (1 - lPrev) * (1 - p.pGuess));
    const lPosterior = softCorrectProb * pObs1 + (1 - softCorrectProb) * pObs0;
    const lNext = lPosterior + (1 - lPosterior) * p.pTransit;
    return Number(clamp(lNext, 0.02, 0.99).toFixed(4));
  }

  public ingestSession(telemetry: SessionLearningTelemetry): LearnerCognitiveTwin {
    if (telemetry.totalStrokes < 1 && telemetry.durationMs < 2000) {
      return this.getTwin();
    }
    const softScore = this.computeSoftObservationScore(telemetry);
    const now = telemetry.capturedAt || Date.now();

    for (const tag of telemetry.knowledgeTags) {
      const existing = this.twin.conceptMastery[tag];
      const prevProb = existing ? existing.masteryProb : DEFAULT_BKT_PARAMS.pInit;
      const nextProb = this.bktUpdateStep(prevProb, softScore);
      const hitPitfall = telemetry.hitPitfalls.length > 0 || telemetry.maxPitfallProb >= 0.68;

      this.twin.conceptMastery[tag] = {
        tag,
        label: KNOWLEDGE_TAG_LABELS[tag],
        masteryProb: nextProb,
        attempts: (existing?.attempts ?? 0) + 1,
        pitfallHits: (existing?.pitfallHits ?? 0) + (hitPitfall ? 1 : 0),
        lastPracticedAt: now,
        streak: softScore >= 0.65 && !hitPitfall ? (existing?.streak ?? 0) + 1 : 0,
        emaAlignment: Number(
          (existing ? 0.72 * existing.emaAlignment + 0.28 * telemetry.meanAlignmentA : telemetry.meanAlignmentA).toFixed(3),
        ),
      };
    }

    const h: WritingHabitMetrics = telemetry.habits;
    const prev = this.twin.habitProfile;
    const alpha = prev.sessionsCount < 4 ? 0.35 : 0.22;
    const hitRate = telemetry.hitPitfalls.length > 0 ? 1 : 0;

    const updatedHabit: WritingHabitProfile = {
      sessionsCount: prev.sessionsCount + 1,
      avgMomentum: Number(((1 - alpha) * prev.avgMomentum + alpha * h.avgMomentumM).toFixed(3)),
      avgEraseRatio: Number(((1 - alpha) * prev.avgEraseRatio + alpha * h.erasePerStrokeRatio).toFixed(3)),
      avgPauseZScore: Number(((1 - alpha) * prev.avgPauseZScore + alpha * h.pauseZScoreMean).toFixed(3)),
      structuralPauseRatio: Number(
        ((1 - alpha) * prev.structuralPauseRatio + alpha * h.structuralBoundaryPauseRatio).toFixed(3),
      ),
      deepThinkingRatio: Number(
        ((1 - alpha) * prev.deepThinkingRatio + alpha * h.deepThinkingHoldRatio).toFixed(3),
      ),
      pitfallFrequency: Number(((1 - alpha) * prev.pitfallFrequency + alpha * hitRate).toFixed(3)),
      dominantWeakness: 'balanced',
    };
    updatedHabit.dominantWeakness = classifyDominantWeakness(updatedHabit);

    this.twin.habitProfile = updatedHabit;
    this.twin.totalSessions += 1;
    if (telemetry.finalStepProgress >= 0.72) {
      this.twin.totalProblemsSolved += 1;
    }
    if (telemetry.hintFeedbackSummary.topEffectiveStyle) {
      this.twin.preferredHintStyle = telemetry.hintFeedbackSummary.topEffectiveStyle;
    }

    this.twin.recentSessions.push({
      sessionId: telemetry.sessionId,
      capturedAt: now,
      problemKey: telemetry.problemKey.slice(0, 64),
      knowledgeTags: telemetry.knowledgeTags,
      finalStepProgress: telemetry.finalStepProgress,
      maxPitfallProb: telemetry.maxPitfallProb,
      erasePerStrokeRatio: h.erasePerStrokeRatio,
      durationMs: telemetry.durationMs,
    });
    if (this.twin.recentSessions.length > 20) {
      this.twin.recentSessions.shift();
    }
    this.twin.updatedAt = now;
    return this.getTwin();
  }

  public getWeakestConcepts(topK = 3): ConceptMasteryState[] {
    return Object.values(this.twin.conceptMastery)
      .sort((a, b) => a.masteryProb - b.masteryProb || b.pitfallHits - a.pitfallHits)
      .slice(0, topK);
  }
}
