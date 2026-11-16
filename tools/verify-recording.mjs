import { readFile } from 'node:fs/promises';

function expectFields(value, fields, location) {
  const isObject = value !== null && typeof value === 'object' && !Array.isArray(value);
  if (!isObject || fields.length !== Object.keys(value).length ||
      !fields.every((key) => Object.hasOwn(value, key))) {
    throw new Error(location + ' 的字段应为 ' + fields.join('、'));
  }
}

function validate(recording) {
  expectFields(recording, ['meta', 'strokes'], '根对象');
  const { meta, strokes } = recording;
  expectFields(meta, ['exportedAt', 'strokeCount', 'pointCount'], 'meta');
  const parsed = Date.parse(meta.exportedAt);
  if (!Number.isFinite(parsed) || new Date(parsed).toISOString() !== meta.exportedAt) {
    throw new Error('exportedAt 应为 UTC ISO 毫秒时间');
  }
  if (!Array.isArray(strokes) || strokes.length !== meta.strokeCount) {
    throw new Error('strokeCount 与实际笔数不符');
  }

  let count = 0;
  let previous = -1;
  const ids = new Set();
  for (const [index, stroke] of strokes.entries()) {
    const label = '第 ' + (index + 1) + ' 笔';
    expectFields(stroke, ['id', 'points'], label);
    if (!Number.isSafeInteger(stroke.id) || stroke.id < 1 || ids.has(stroke.id)) {
      throw new Error(label + ' 的 id 应为唯一正整数');
    }
    ids.add(stroke.id);
    if (!Array.isArray(stroke.points) || !stroke.points.length) throw new Error(label + ' 缺少触点');
    for (const [i, point] of stroke.points.entries()) {
      const where = label + '第 ' + (i + 1) + ' 点';
      expectFields(point, ['x', 'y', 't'], where);
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) throw new Error(where + ' 坐标无效');
      if (!Number.isSafeInteger(point.t) || point.t <= previous) {
        throw new Error(where + ' 的时间戳须为非负整数，且大于上一触点（含跨笔）');
      }
      previous = point.t;
    }
    count += stroke.points.length;
  }
  if (count !== meta.pointCount) throw new Error('pointCount 与实际总点数不符');
}

function windowReport(points) {
  const first = points[0].t;
  const last = points[points.length - 1].t;
  let time = first + 1000;
  let entered = 0;
  let expired = 0;
  let minimum = Infinity;
  let minimumAt = time;
  let maximum = 0;
  // 以整数毫秒定义窗口 (time-1000, time]。计数仅在点进入或过期时变化，
  // 因此检查每个变点等价于检查每一毫秒，可发现恰好落在两次采样之间的短暂下降。
  while (time <= last) {
    while (entered < points.length && points[entered].t <= time) entered++;
    while (expired < points.length && points[expired].t <= time - 1000) expired++;
    const rate = entered - expired;
    if (rate < minimum) {
      minimum = rate;
      minimumAt = time;
    }
    maximum = Math.max(maximum, rate);
    const nextEntry = points[entered]?.t ?? Infinity;
    const nextExpiry = (points[expired]?.t ?? Infinity) + 1000;
    time = Math.min(nextEntry, nextExpiry);
  }
  return { minimum, maximum, atSeconds: (minimumAt - first) / 1000 };
}

async function main() {
  if (process.argv.length !== 3) {
    console.error('用法：node tools/verify-recording.mjs <导出的文件.json>');
    return 64;
  }
  const data = JSON.parse((await readFile(process.argv[2], 'utf8')).replace(/^\uFEFF/, ''));
  validate(data);
  console.log('格式及全局时序通过。' + data.meta.strokeCount + ' 笔，' + data.meta.pointCount + ' 点。');
  const longStrokes = data.strokes.filter(({ points }) => points.at(-1).t - points[0].t >= 10_000);
  if (!longStrokes.length) {
    console.log('未能判定采样率。请持续划线至少 10 秒后抬笔并重新导出。');
    return 2;
  }

  let passed = true;
  for (const { id, points } of longStrokes) {
    const { minimum, maximum, atSeconds } = windowReport(points);
    const duration = (points.at(-1).t - points[0].t) / 1000;
    console.log('笔 ' + id + '，时长 ' + duration.toFixed(3) + ' 秒，完整 1 秒窗口 ' +
      minimum + '–' + maximum + ' 点/秒，最低值首次出现于落笔后 ' + atSeconds.toFixed(3) + ' 秒。');
    if (minimum < 60) passed = false;
  }
  console.log(passed ? '采样率达标，受检长笔的全部完整窗口均 ≥ 60 点/秒。' : '采样率未达标，存在低于 60 点/秒的完整窗口。');
  return passed ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (error) {
  console.error('验收失败。' + (error instanceof Error ? error.message : String(error)));
  process.exitCode = 1;
}
