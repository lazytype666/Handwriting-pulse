export type StallType = 'pause_erase' | 'speed_drop';
export type InkTool = 'pen' | 'eraser';

export interface StallEvent {
  id: number;
  type: StallType;
  startedAt: number;
  x: number;
  y: number;
  durationMs: number;
}

export interface Thresholds {
  stallMs: number;
  speedRatio: number;
  cooldownMs: number;
}

export const DEFAULT_THRESHOLDS: Thresholds = {
  stallMs: 3000, speedRatio: 0.3, cooldownMs: 10_000,
};

export const SPEED_WINDOW_MS = 1000;
export const BASELINE_WINDOW_MS = 30_000;
export const REWRITE_WINDOW_MS = 3000;
export const ERASER_WIDTH = 26;
export const REWRITE_RADIUS = ERASER_WIDTH / 2 + 8;
// 持续减速才触发，避免单次不均匀送达造成误报。
export const SLOW_CONFIRM_MS = 220;
