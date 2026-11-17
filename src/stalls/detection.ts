import {
  REWRITE_WINDOW_MS, SLOW_CONFIRM_MS, SPEED_WINDOW_MS,
  type StallEvent, type StallType, type Thresholds,
} from './config';
import { type Features, speedRatioAt } from './features';

export function createDetector() {
  return { events: [] as StallEvent[], lastAt: -1, slowSince: -1, slowTriggered: false };
}

type Detector = ReturnType<typeof createDetector>;

export function cooldownLeft(d: Detector, now: number, limits: Thresholds) {
  'worklet';
  return d.lastAt < 0 ? 0 : Math.max(0, d.lastAt + limits.cooldownMs - now);
}

// 判定只由真实触点驱动。空闲帧可以显示停顿，但不能把抬笔想题误判为“正在减速书写”。
export function detect(
  d: Detector, s: Features, rewrite: boolean, penMoving: boolean,
  now: number, epoch: number, x: number, y: number, limits: Thresholds,
): StallEvent | null {
  'worklet';
  let type: StallType | null = null;
  let duration = 0;
  const ratio = penMoving && s.penRunStart >= 0 && now - s.penRunStart >= SPEED_WINDOW_MS
    ? speedRatioAt(s, now) : null;

  if (ratio === null || ratio >= limits.speedRatio) {
    d.slowSince = -1;
    d.slowTriggered = false;
  } else if (d.slowSince < 0) {
    d.slowSince = now;
  } else if (!d.slowTriggered && now - d.slowSince >= SLOW_CONFIRM_MS) {
    type = 'speed_drop';
    duration = now - d.slowSince;
  }
  if (rewrite && s.pauseEndedAt >= 0 && now - s.pauseEndedAt <= REWRITE_WINDOW_MS && s.pauseDuration >= limits.stallMs) {
    type = 'pause_erase';
    duration = s.pauseDuration;
    s.pauseEndedAt = -1;
  }
  if (!type || cooldownLeft(d, now, limits) > 0) return null;
  if (type === 'speed_drop') d.slowTriggered = true;
  const event = { id: d.events.length + 1, type, startedAt: epoch, x, y, durationMs: Math.round(duration) };
  d.events.push(event);
  d.lastAt = now;
  return event;
}
