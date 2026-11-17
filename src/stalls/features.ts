import {
  BASELINE_WINDOW_MS, REWRITE_RADIUS, REWRITE_WINDOW_MS, SPEED_WINDOW_MS,
  type InkTool, type Thresholds,
} from './config';

interface Position { x: number; y: number }
interface LengthSample { time: number; total: number }

export function createFeatures() {
  return {
    firstPenAt: -1, lastTouch: -1, lastX: 0, lastY: 0, penRunStart: -1,
    pauseEndedAt: -1, pauseDuration: 0, eraseAt: -1,
    erasePath: [] as Position[], rewrittenAt: -1,
    total: 0, samples: [] as LengthSample[],
  };
}

export type Features = ReturnType<typeof createFeatures>;

// 自研空间特征：到橡皮轨迹折线的最近距离。即使移动事件稀疏，线段中部擦过的位置也能命中。
function nearErasure(path: Position[], x: number, y: number) {
  'worklet';
  for (let i = 0; i < path.length; i++) {
    const a = path[Math.max(0, i - 1)];
    const b = path[i];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const square = dx * dx + dy * dy;
    const k = square ? Math.max(0, Math.min(1, ((x - a.x) * dx + (y - a.y) * dy) / square)) : 0;
    if (Math.hypot(x - a.x - k * dx, y - a.y - k * dy) <= REWRITE_RADIUS) return true;
  }
  return false;
}

// 自研累计弧长序列。仅累计笔尖在同一笔中的移动，抬笔跳到另一处与橡皮移动均不计长度。
export function observeTouch(s: Features, tool: InkTool, down: boolean, x: number, y: number, now: number, limits: Thresholds) {
  'worklet';
  const gap = s.lastTouch < 0 ? 0 : now - s.lastTouch;
  if (s.lastTouch >= 0 && gap >= limits.stallMs) {
    s.pauseDuration = gap;
    s.pauseEndedAt = now;
  }

  let rewrite = false;
  if (tool === 'eraser') {
    s.penRunStart = -1;
    if (down) s.erasePath = [];
    s.erasePath.push({ x, y });
    s.eraseAt = now;
  } else {
    if (s.firstPenAt < 0) s.firstPenAt = now;
    if (s.penRunStart < 0 || gap >= SPEED_WINDOW_MS) s.penRunStart = now;
    if (down) {
      rewrite = s.eraseAt >= 0 && now - s.eraseAt <= REWRITE_WINDOW_MS && nearErasure(s.erasePath, x, y);
      if (rewrite) s.rewrittenAt = now;
      s.erasePath = [];
      s.eraseAt = -1;
    }
    if (!down) s.total += Math.hypot(x - s.lastX, y - s.lastY);
    s.samples.push({ time: now, total: s.total });
    // 分批清理，保留窗口左侧一个样本用于线性插值。避免逐点移动整个数组。
    if (s.samples.length % 128 === 0) {
      let expired = 0;
      while (expired + 1 < s.samples.length && s.samples[expired + 1].time < now - BASELINE_WINDOW_MS) expired++;
      if (expired) s.samples.splice(0, expired);
    }
  }
  s.lastX = x;
  s.lastY = y;
  s.lastTouch = now;
  return rewrite;
}

function lengthAt(samples: LengthSample[], time: number) {
  'worklet';
  let low = 0;
  let high = samples.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (samples[mid].time <= time) low = mid + 1;
    else high = mid;
  }
  if (low === 0) return samples[0].total;
  const a = samples[low - 1];
  if (low === samples.length) return a.total;
  const b = samples[low];
  return a.total + (b.total - a.total) * ((time - a.time) / (b.time - a.time));
}

export function speedRatioAt(s: Features, now: number): number | null {
  'worklet';
  const span = Math.min(BASELINE_WINDOW_MS, now - s.firstPenAt);
  if (s.firstPenAt < 0 || span < SPEED_WINDOW_MS || s.samples.length < 2) return null;
  const baselineLength = s.total - lengthAt(s.samples, now - span);
  if (baselineLength <= 0) return null;
  const recentLength = s.total - lengthAt(s.samples, now - SPEED_WINDOW_MS);
  return (recentLength / SPEED_WINDOW_MS) / (baselineLength / span);
}

export function pauseAt(s: Features, now: number) {
  'worklet';
  return s.lastTouch < 0 ? 0 : Math.max(0, now - s.lastTouch);
}
