export interface Point {
  x: number;
  y: number;
  t: number;
}

export interface Stroke {
  id: number;
  points: Point[];
}

export interface Diagnostics {
  rate: number;
  delta: number | null;
}

export interface Recorder {
  strokes: Stroke[];
  writer: number | null;
  epochOffset: number | null;
  lastTimestamp: number;
  recent: number[];
  delta: number | null;
}

export function createRecorder(): Recorder {
  return {
    strokes: [], writer: null, epochOffset: null, lastTimestamp: 0,
    recent: [], delta: null,
  };
}

// 自研采集器。调用方只传入当前执笔触点，落笔时分组，移动与抬笔只追加。
// 不按帧补点，也不因两次坐标相同而丢掉真实回调。
export function beginStroke(state: Recorder, pointerId: number): void {
  'worklet';
  state.writer = pointerId;
  state.strokes.push({ id: state.strokes.length + 1, points: [] });
}

export function appendPoint(state: Recorder, x: number, y: number, now: number): void {
  'worklet';
  // RNGH 原始 changedTouches 仅保证坐标和触点编号，没有可移植的硬件时间戳。
  // now 在 UI 回调入口用 performance.now() 取得，表示送达时间。仅首次映射到 Unix 毫秒，
  // 防止后续系统校时造成倒退。同一整数毫秒内的多点以相邻整数编码以保留先后顺序。
  if (state.epochOffset === null) state.epochOffset = Date.now() - now;
  const t = Math.max(state.lastTimestamp + 1, Math.round(state.epochOffset + now));
  state.strokes[state.strokes.length - 1].points.push({ x, y, t });
  state.delta = state.lastTimestamp === 0 ? null : t - state.lastTimestamp;
  state.lastTimestamp = t;
  state.recent.push(now);
}

// 使用未经整数修正的到达时间计算 (now - 1000, now] 内真实点数。
// 显示节流与写点互相独立，停止书写后一秒内旧数据自然离窗。
export function readDiagnostics(state: Recorder, now: number): Diagnostics {
  'worklet';
  const cutoff = now - 1000;
  let expired = 0;
  while (expired < state.recent.length && state.recent[expired] <= cutoff) expired++;
  if (expired > 0) state.recent.splice(0, expired);
  return { rate: state.recent.length, delta: state.delta };
}

// 只在用户导出时复制。采集和复制都在 UI 线程串行执行，因此可包含正在写的笔，
// 快照建立后的移动不会修改之前导出的数组。
export function copyStrokes(state: Recorder): Stroke[] {
  'worklet';
  return state.strokes.map(stroke => ({
    id: stroke.id,
    points: stroke.points.map(point => ({ x: point.x, y: point.y, t: point.t })),
  }));
}
