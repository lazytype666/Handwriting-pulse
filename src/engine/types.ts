import type { Diagnostics, Stroke } from '../capture';
import type { HintLevel } from '../hints/safety';
import type { StallEvent, StallType, Thresholds } from '../stalls';
import type { CognitiveState, CompanionMode, MotorFingerprint } from './cognitiveState';
import type { FeedbackLogEntry, FeedbackSignal, HintStyle, PolicySnapshot } from './personalizedPolicy';
import type { SolutionCacheEntry } from './solutionCache';

/**
 * 端侧智能引擎标准契约接口（预留给端侧模型接入及前后端/数据库协作）。
 */

export interface IProblemIngestorAdapter {
  ingestFromText(rawText: string): Promise<{ problemId: string; problemText: string }>;
  ingestFromImageUri?(imageUri: string): Promise<{ problemId: string; problemText: string }>;
}

export interface IInkSyntaxEncoderAdapter {
  isLoaded(): boolean;
  encodeRecentStrokes(strokes: Stroke[]): Promise<{
    recognizedLatex: string;
    alignmentScore: number;
  }>;
}

export interface IEdgeHintModelAdapter {
  isReady(): boolean;
  generateHint(params: {
    question: string;
    type: StallType;
    level: HintLevel;
    durationMs: number;
    style: HintStyle;
  }): Promise<string>;
}

export interface IStorageAdapter {
  saveSession(payload: EngineSessionPayload): Promise<void>;
  loadUserProfile(userId: string): Promise<{
    fingerprint: MotorFingerprint;
    policy: PolicySnapshot;
  } | null>;
  saveUserProfile(
    userId: string,
    profile: { fingerprint: MotorFingerprint; policy: PolicySnapshot },
  ): Promise<void>;
}

export interface EngineSessionPayload {
  exportedAt: string;
  problemText: string;
  strokes: Stroke[];
  stallEvents: StallEvent[];
  cognitiveSummary: {
    companionMode: CompanionMode;
    interventionThreshold: number;
    fingerprint: MotorFingerprint;
    policySnapshot: PolicySnapshot;
    latestState: CognitiveState;
    feedbackLogs: FeedbackLogEntry[];
    cachedProblems: SolutionCacheEntry[];
  };
}

export interface EngineDiagnosticsTick {
  cognitiveState: CognitiveState;
  interventionThreshold: number;
  cognitiveStallEvent: StallEvent | null;
  settledFeedback: FeedbackLogEntry | null;
}

export interface IHandwritingPulseEngine {
  updateProblemText(problemText: string): Promise<boolean>;
  setCompanionMode(mode: CompanionMode): void;
  setSemanticAlignment(alignmentA: number): void;
  processDiagnosticsTick(
    diagnostics: Diagnostics,
    thresholds: Thresholds,
    nowMs?: number,
  ): EngineDiagnosticsTick;
  selectStyleForStall(type: StallType): HintStyle;
  recordExplicitFeedback(action: 'request_more' | 'dismiss' | 'thinking'): FeedbackLogEntry | null;
  registerEdgeModelAdapter(adapter: IEdgeHintModelAdapter | null): void;
}

export type { FeedbackSignal, HintStyle, PolicySnapshot };
