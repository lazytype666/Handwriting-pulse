import type { Point, Stroke } from '../capture';
import { REWRITE_RADIUS } from '../stalls/config';
import type { GraphGeometricRole } from './graphLogits';
import {
  MATH_VOCAB_INDEX,
  MATH_VOCAB_SIZE,
  type CtcDecodeOptions,
} from './mathVocab';
import {
  buildProblemVocabPriorBias,
  EdgeNeuralModelRunner,
  extractPairGeom10,
} from './neuralRuntime';
import type { SolutionCacheEntry, SolutionPitfallSpec } from './solutionCache';
import type { IInkSyntaxEncoderAdapter } from './types';

let sharedNeuralRunner: EdgeNeuralModelRunner | null = null;

/** 获取全局复用的纯 TS 端侧双流神经网络运行时实例（只解码一次量化权重） */
export function getSharedNeuralRunner(): EdgeNeuralModelRunner {
  if (!sharedNeuralRunner) {
    sharedNeuralRunner = new EdgeNeuralModelRunner();
  }
  return sharedNeuralRunner;
}

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** 解耦空间几何特征向量：与 Google MathWriting 五通道输入对齐 */
export interface SpatialSamplePoint {
  dx: number;
  dy: number;
  cosTheta: number;
  sinTheta: number;
  penState: 0 | 1;
}

export type StructuralRole =
  | 'logical_connector'
  | 'fraction_bar'
  | 'equation_rel'
  | 'calculus_operator'
  | 'matrix_delimiter'
  | 'probability_rel'
  | 'operand'
  | 'scratch_draft';

export type MathDomainHint = 'calculus' | 'linear_algebra' | 'probability' | 'general';

export interface StrokeSymbolToken {
  token: string;
  confidence: number;
  role: StructuralRole;
  domain: MathDomainHint;
  bbox: BoundingBox;
  strokeIds: number[];
}

export interface EncodedExpressionResult {
  recognizedLatex: string;
  structuralRole: StructuralRole;
  detectedDomain: MathDomainHint;
  structuralBoundaryFactor: number;
  alignmentScore: number;
  matchedStepIndex: number;
  hitPitfall: SolutionPitfallSpec | null;
  /** 图二：解题图谱连续易错歧路触发概率 PitfallProb ∈ [0, 1] */
  pitfallProb: number;
  /** 图四：期待步骤关键锚点缺失缺口 AnchorGap ∈ [0, 1] */
  anchorGap: number;
  /** 图四：解题步骤归一化进度 StepProgress ∈ [0, 1] */
  stepProgress: number;
  /** 125 维融合边缘后验概率（CTC + 几何观测融合） */
  tokenPosteriors?: Float32Array;
  activeLineCount: number;
  tokenCount: number;
  encoderMode: 'geometric_heuristic' | 'neural_quantized_ts' | 'neural_onnx';
}

interface StrokeMetrics {
  stroke: Stroke;
  bbox: BoundingBox;
  cx: number;
  cy: number;
  length: number;
  horizontalness: number;
  verticalness: number;
  closureRatio: number;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  diagSlope: number;
  directionFlipsX: number;
  directionFlipsY: number;
  chordDeviation: number;
  maxY: number;
  minY: number;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

export function computeBoundingBox(points: Point[]): BoundingBox {
  if (points.length === 0) return { x: 0, y: 0, w: 0, h: 0 };
  let minX = points[0].x;
  let maxX = points[0].x;
  let minY = points[0].y;
  let maxY = points[0].y;
  for (let i = 1; i < points.length; i++) {
    const p = points[i];
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return {
    x: minX,
    y: minY,
    w: Math.max(1, maxX - minX),
    h: Math.max(1, maxY - minY),
  };
}

/**
 * 提取与时间彻底解耦的 5 通道几何空间流（复刻 MathWriting 预处理协议）。
 * 【自研实现】去除时间戳 t，通过步长尺度归一化确保快慢手写的相同公式共享几何流形。
 */
export function extractSpatialStream(strokes: Stroke[]): SpatialSamplePoint[] {
  const flat: SpatialSamplePoint[] = [];
  if (strokes.length === 0) return flat;

  let minY = Infinity;
  let maxY = -Infinity;
  for (const s of strokes) {
    if (s.tool === 'eraser') continue;
    for (const p of s.points) {
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
  }
  const scale = Math.max(20, maxY - minY);

  for (const s of strokes) {
    if (s.tool === 'eraser' || s.points.length === 0) continue;
    for (let i = 0; i < s.points.length; i++) {
      const curr = s.points[i];
      const prev = i > 0 ? s.points[i - 1] : curr;
      const dxRaw = (curr.x - prev.x) / scale;
      const dyRaw = (curr.y - prev.y) / scale;
      const ds = Math.hypot(dxRaw, dyRaw);
      const cosTheta = ds > 1e-6 ? dxRaw / ds : 1;
      const sinTheta = ds > 1e-6 ? dyRaw / ds : 0;
      const isLast = i === s.points.length - 1;
      flat.push({
        dx: Number(dxRaw.toFixed(4)),
        dy: Number(dyRaw.toFixed(4)),
        cosTheta: Number(cosTheta.toFixed(4)),
        sinTheta: Number(sinTheta.toFixed(4)),
        penState: isLast ? 0 : 1,
      });
    }
  }
  return flat;
}

/** 过滤被后续橡皮擦轨迹覆盖擦除的旧笔画，仅保留当前纸面有效墨迹 */
function filterVisiblePenStrokes(strokes: Stroke[]): Stroke[] {
  const visible: Stroke[] = [];
  const eraserHistory: { strokeIndex: number; points: Point[] }[] = [];

  for (let i = 0; i < strokes.length; i++) {
    if (strokes[i].tool === 'eraser') {
      eraserHistory.push({ strokeIndex: i, points: strokes[i].points });
    }
  }

  for (let i = 0; i < strokes.length; i++) {
    const s = strokes[i];
    if (s.tool === 'eraser' || s.points.length === 0) continue;

    let erasedPoints = 0;
    const step = Math.max(1, Math.floor(s.points.length / 8));
    let checked = 0;

    for (let pIdx = 0; pIdx < s.points.length; pIdx += step) {
      checked += 1;
      const pt = s.points[pIdx];
      let hit = false;
      for (const er of eraserHistory) {
        if (er.strokeIndex < i) continue;
        for (let k = 0; k < er.points.length; k++) {
          if (Math.hypot(pt.x - er.points[k].x, pt.y - er.points[k].y) <= REWRITE_RADIUS) {
            hit = true;
            break;
          }
        }
        if (hit) break;
      }
      if (hit) erasedPoints += 1;
    }

    if (checked > 0 && erasedPoints / checked >= 0.55) {
      continue;
    }
    visible.push(s);
  }
  return visible;
}

function measureStroke(stroke: Stroke): StrokeMetrics {
  const pts = stroke.points;
  const bbox = computeBoundingBox(pts);
  let length = 0;
  let flipsX = 0;
  let flipsY = 0;
  let prevSignX = 0;
  let prevSignY = 0;

  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i].x - pts[i - 1].x;
    const dy = pts[i].y - pts[i - 1].y;
    length += Math.hypot(dx, dy);
    if (Math.abs(dx) > 1.5) {
      const s = dx > 0 ? 1 : -1;
      if (prevSignX !== 0 && s !== prevSignX) flipsX += 1;
      prevSignX = s;
    }
    if (Math.abs(dy) > 1.5) {
      const s = dy > 0 ? 1 : -1;
      if (prevSignY !== 0 && s !== prevSignY) flipsY += 1;
      prevSignY = s;
    }
  }

  const start = pts[0];
  const end = pts[pts.length - 1];
  const netDx = Math.abs(end.x - start.x);
  const netDy = Math.abs(end.y - start.y);
  const chordDx = end.x - start.x;
  const chordDy = end.y - start.y;
  const chordLen = Math.hypot(chordDx, chordDy);
  let maxPerp = 0;
  if (chordLen > 4) {
    for (let i = 1; i < pts.length - 1; i++) {
      const dist =
        Math.abs((pts[i].x - start.x) * chordDy - (pts[i].y - start.y) * chordDx) / chordLen;
      if (dist > maxPerp) maxPerp = dist;
    }
  }
  const chordDeviation = maxPerp / Math.max(1, length);

  const diag = Math.hypot(bbox.w, bbox.h);
  const closureRatio = diag > 8 ? chordLen / diag : 1;

  return {
    stroke,
    bbox,
    cx: bbox.x + bbox.w / 2,
    cy: bbox.y + bbox.h / 2,
    length,
    horizontalness: netDx / Math.max(1, netDx + netDy),
    verticalness: netDy / Math.max(1, netDx + netDy),
    closureRatio,
    startX: start.x,
    startY: start.y,
    endX: end.x,
    endY: end.y,
    diagSlope: (end.y - start.y) / Math.max(1, Math.abs(end.x - start.x)),
    directionFlipsX: flipsX,
    directionFlipsY: flipsY,
    chordDeviation,
    minY: bbox.y,
    maxY: bbox.y + bbox.h,
  };
}

function xOverlapRatio(a: BoundingBox, b: BoundingBox): number {
  const left = Math.max(a.x, b.x);
  const right = Math.min(a.x + a.w, b.x + b.w);
  const inter = Math.max(0, right - left);
  return inter / Math.max(1, Math.min(a.w, b.w));
}

function yOverlapRatio(a: BoundingBox, b: BoundingBox): number {
  const top = Math.max(a.y, b.y);
  const bottom = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, bottom - top);
  return inter / Math.max(1, Math.min(a.h, b.h));
}

function mergeBbox(list: BoundingBox[]): BoundingBox {
  let minX = list[0].x;
  let minY = list[0].y;
  let maxX = list[0].x + list[0].w;
  let maxY = list[0].y + list[0].h;
  for (let i = 1; i < list.length; i++) {
    minX = Math.min(minX, list[i].x);
    minY = Math.min(minY, list[i].y);
    maxX = Math.max(maxX, list[i].x + list[i].w);
    maxY = Math.max(maxY, list[i].y + list[i].h);
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

/**
 * 识别大学数学全领域常见符号原语（高数 / 线代 / 概率论 / 基础代数）。
 */
function classifyCluster(cluster: StrokeMetrics[], lineHeight: number): StrokeSymbolToken {
  const bbox = mergeBbox(cluster.map(m => m.bbox));
  const strokeIds = cluster.map(m => m.stroke.id);

  // 1. 三笔画及以上特殊复合结构识别：恒等号 '\equiv'、积分/闭环积分、矩阵连乘 '\prod'、箭头 '\Rightarrow'
  if (cluster.length === 3) {
    const allHoriz = cluster.every(m => m.horizontalness > 0.72);
    if (
      allHoriz &&
      xOverlapRatio(cluster[0].bbox, cluster[1].bbox) > 0.45 &&
      xOverlapRatio(cluster[1].bbox, cluster[2].bbox) > 0.45
    ) {
      return { token: '\\equiv', confidence: 0.88, role: 'equation_rel', domain: 'general', bbox, strokeIds };
    }
    // 双横线 + 箭头折钩 => '\Rightarrow'
    const twoHoriz = cluster.filter(m => m.horizontalness > 0.75);
    if (twoHoriz.length >= 2 && bbox.w > bbox.h * 1.1) {
      return { token: '\\Rightarrow', confidence: 0.82, role: 'logical_connector', domain: 'general', bbox, strokeIds };
    }
    // 顶部横线 + 双竖腿 => '\prod' 或 '\pi'
    const horiz = cluster.find(m => m.horizontalness > 0.72);
    const verts = cluster.filter(m => m.verticalness > 0.68);
    if (horiz && verts.length === 2 && horiz.cy < (verts[0].cy + verts[1].cy) / 2) {
      const tok = bbox.h > lineHeight * 1.2 ? '\\prod' : '\\pi';
      return {
        token: tok,
        confidence: 0.8,
        role: tok === '\\prod' ? 'calculus_operator' : 'operand',
        domain: tok === '\\prod' ? 'calculus' : 'general',
        bbox,
        strokeIds,
      };
    }
    // 一纵拐三角 / 梯度 '\nabla'（三边首尾大致相接）
    const totalLen = cluster.reduce((s, m) => s + m.length, 0);
    if (totalLen > 36 && bbox.w >= 14 && bbox.h >= 14 && horiz && horiz.cy < bbox.y + bbox.h * 0.45) {
      return { token: '\\nabla', confidence: 0.74, role: 'calculus_operator', domain: 'calculus', bbox, strokeIds };
    }
  }

  // 2. 双笔画组合：'='、'+'、'\times/x'、'\Phi'、'\to'、'\le/\ge'、'\neq'、'\oint'、'\varnothing'
  if (cluster.length === 2) {
    const [a, b] = cluster;
    // 平行双横 -> '=' 或 '\approx'
    if (a.horizontalness > 0.72 && b.horizontalness > 0.72 && xOverlapRatio(a.bbox, b.bbox) > 0.45) {
      const verticalGap = Math.abs(a.cy - b.cy);
      if (verticalGap >= 3 && verticalGap <= Math.max(26, lineHeight * 0.85)) {
        const wavy = a.directionFlipsY >= 2 || b.directionFlipsY >= 2;
        return {
          token: wavy ? '\\approx' : '=',
          confidence: 0.9,
          role: 'equation_rel',
          domain: 'general',
          bbox,
          strokeIds,
        };
      }
    }
    // 正交十字 -> '+' 或 闭合环被竖线贯穿 -> '\Phi'（概率论标准正态分布函数）
    const loopStroke = [a, b].find(m => m.closureRatio < 0.36 && m.length > 24);
    const vertStroke = [a, b].find(m => m.verticalness > 0.72 && m.bbox.h >= 14);
    if (loopStroke && vertStroke && xOverlapRatio(loopStroke.bbox, vertStroke.bbox) > 0.2) {
      return { token: '\\Phi', confidence: 0.82, role: 'operand', domain: 'probability', bbox, strokeIds };
    }
    // 闭合环被斜线贯穿 -> '\varnothing'（空集）或 积分闭环 '\oint'
    const slashStroke = [a, b].find(m => Math.abs(m.diagSlope) > 0.45 && m.length > 16);
    if (loopStroke && slashStroke && slashStroke !== loopStroke) {
      const isTallInt = slashStroke.bbox.h >= Math.max(36, lineHeight * 1.45);
      return {
        token: isTallInt ? '\\oint' : '\\varnothing',
        confidence: 0.78,
        role: isTallInt ? 'calculus_operator' : 'operand',
        domain: isTallInt ? 'calculus' : 'probability',
        bbox,
        strokeIds,
      };
    }

    const oneHorizOneVert =
      (a.horizontalness > 0.68 && b.verticalness > 0.68) ||
      (b.horizontalness > 0.68 && a.verticalness > 0.68);
    if (oneHorizOneVert) {
      return { token: '+', confidence: 0.88, role: 'logical_connector', domain: 'general', bbox, strokeIds };
    }
    // 右向水平线 + 右端折叠箭头 -> '\to'（趋近于，如 x -> 0）
    const hArrow = [a, b].find(m => m.horizontalness > 0.75 && m.bbox.w >= 16);
    const head = [a, b].find(m => m !== hArrow && m.cx > (hArrow?.bbox.x ?? 0) + (hArrow?.bbox.w ?? 0) * 0.55);
    if (hArrow && head && head.bbox.w < hArrow.bbox.w * 0.75) {
      return { token: '\\to', confidence: 0.84, role: 'logical_connector', domain: 'calculus', bbox, strokeIds };
    }
    if (a.diagSlope * b.diagSlope < -0.25 && Math.abs(a.cx - b.cx) < Math.max(18, bbox.w * 0.45)) {
      return { token: 'x', confidence: 0.8, role: 'operand', domain: 'general', bbox, strokeIds };
    }
  }

  // 3. 单笔画原语：涵盖积分号 '\int'、求和号 '\sum'、无穷 '\infty'、分布波浪 '~'、交并集 '\cap/\cup'、偏导 '\partial'、分式线、界定符等
  if (cluster.length === 1) {
    const m = cluster[0];
    // (a) 积分号 '\int'：大高度 S 型弯钩（高宽比高且含上下弯钩弦偏离或 X 方向微回勾）
    if (
      m.bbox.h >= Math.max(38, lineHeight * 1.25) &&
      m.verticalness > 0.6 &&
      (m.directionFlipsX >= 1 || m.chordDeviation >= 0.055)
    ) {
      return {
        token: '\\int',
        confidence: 0.85,
        role: 'calculus_operator',
        domain: 'calculus',
        bbox,
        strokeIds,
      };
    }
    // (b) 行列式/矩阵高竖线定界符 或 条件概率 '|'（笔直大竖线，弦偏离极小且窄幅垂直）
    if (
      m.chordDeviation < 0.055 &&
      m.verticalness > 0.78 &&
      m.bbox.h >= Math.max(40, lineHeight * 1.55) &&
      m.bbox.w <= m.bbox.h * 0.25
    ) {
      return {
        token: '|',
        confidence: 0.86,
        role: 'matrix_delimiter',
        domain: 'linear_algebra',
        bbox,
        strokeIds,
      };
    }
    // (c) 概率论服从分布波浪 '~'：横向展开且垂直起伏翻转 >= 2
    if (m.horizontalness > 0.65 && m.directionFlipsY >= 2 && m.bbox.w >= 14 && m.bbox.h < lineHeight * 0.8) {
      return {
        token: '~',
        confidence: 0.82,
        role: 'probability_rel',
        domain: 'probability',
        bbox,
        strokeIds,
      };
    }
    // (d) 横杠 / 分式线 '\frac'
    if (m.horizontalness > 0.78 && m.bbox.w >= 10 && m.bbox.h <= Math.max(16, m.bbox.w * 0.45)) {
      const role: StructuralRole = m.bbox.w >= 46 ? 'fraction_bar' : 'logical_connector';
      return {
        token: m.bbox.w >= 46 ? '\\frac' : '-',
        confidence: 0.84,
        role,
        domain: 'general',
        bbox,
        strokeIds,
      };
    }
    // (e) 普通竖线 / 数字 '1' / 条件竖杠
    if (m.verticalness > 0.76 && m.bbox.h >= 12 && m.bbox.w <= m.bbox.h * 0.48) {
      return { token: '1', confidence: 0.75, role: 'operand', domain: 'general', bbox, strokeIds };
    }
    // (f) 求和符号 '\sum'（单笔 4 折 Z/Sigma 折线，X 至少往返翻转 2 次且有足够纵深）
    if (m.directionFlipsX >= 2 && m.directionFlipsY >= 1 && m.bbox.h >= lineHeight * 0.85 && m.closureRatio > 0.36) {
      return {
        token: '\\sum',
        confidence: 0.8,
        role: 'calculus_operator',
        domain: 'calculus',
        bbox,
        strokeIds,
      };
    }
    // (g) 无穷大 '\infty'（横宽闭合双环：长径比宽 + 首尾近闭 + X/Y 多回转）
    if (m.closureRatio < 0.38 && m.bbox.w > m.bbox.h * 1.45 && m.directionFlipsX >= 1 && m.directionFlipsY >= 2) {
      return {
        token: '\\infty',
        confidence: 0.82,
        role: 'operand',
        domain: 'calculus',
        bbox,
        strokeIds,
      };
    }
    // (h) 偏导 '\partial'（中下闭合但顶部留高弧尾）或 闭合环 '0'
    if (m.closureRatio < 0.35 && m.length > 24) {
      if (m.bbox.h > m.bbox.w * 1.35 && m.startY < m.cy - m.bbox.h * 0.25) {
        return { token: '\\partial', confidence: 0.74, role: 'calculus_operator', domain: 'calculus', bbox, strokeIds };
      }
      return { token: '0', confidence: 0.78, role: 'operand', domain: 'general', bbox, strokeIds };
    }
    // (i) 集合交/并 '\cap' / '\cup' 或 括号 '(' / ')'
    if (m.horizontalness > 0.42 && m.directionFlipsY === 1 && m.bbox.w > m.bbox.h * 0.9) {
      const midPt = m.stroke.points[Math.floor(m.stroke.points.length / 2)];
      const tok = midPt.y < m.startY && midPt.y < m.endY ? '\\cap' : '\\cup';
      return { token: tok, confidence: 0.75, role: 'probability_rel', domain: 'probability', bbox, strokeIds };
    }
    if (m.verticalness > 0.55 && m.directionFlipsX === 1 && m.bbox.h > m.bbox.w * 1.6) {
      const midPt = m.stroke.points[Math.floor(m.stroke.points.length / 2)];
      const tok = midPt.x < m.startX ? '(' : ')';
      const role: StructuralRole = m.bbox.h >= lineHeight * 1.6 ? 'matrix_delimiter' : 'operand';
      return {
        token: tok,
        confidence: 0.76,
        role,
        domain: role === 'matrix_delimiter' ? 'linear_algebra' : 'general',
        bbox,
        strokeIds,
      };
    }
    // (j) 根号 '\sqrt'
    if (m.bbox.w >= 26 && m.bbox.h >= 16 && m.stroke.points.length >= 6) {
      const pts = m.stroke.points;
      const mid = pts[Math.floor(pts.length * 0.45)];
      if (mid.y > m.startY + m.bbox.h * 0.35 && m.endX > m.startX + m.bbox.w * 0.5) {
        return { token: '\\sqrt', confidence: 0.72, role: 'logical_connector', domain: 'general', bbox, strokeIds };
      }
    }
  }

  // 多笔画字母 / 数字 / 矩阵元素 / 希腊字母（如 λ, α, E(X), P(A), dx）
  const avgLen = cluster.reduce((sum, m) => sum + m.length, 0);
  if (avgLen < 12) {
    return { token: '·', confidence: 0.5, role: 'scratch_draft', domain: 'general', bbox, strokeIds };
  }
  return {
    token: cluster.length >= 3 ? 'expr' : 'sym',
    confidence: 0.65,
    role: 'operand',
    domain: 'general',
    bbox,
    strokeIds,
  };
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-clamp(x, -30, 30)));
}

/**
 * Quote ① 二维拓扑规范化重排（Spatial Canonical Stroke Reordering）：
 * 在送入 Conformer-CTC 前解除“回头补写分子、补写积分上限或局部修改”导致的时间乱序：
 * 1) 检测宽水平分式线，归组为 [分子笔画] -> [分式线] -> [分母笔画]；
 * 2) 检测高大竖直微积分主算符（如 ∫/∑），将贴靠其上下端的界限簇排序至主体之后；
 * 3) 其余同级笔画按中心横坐标 x 从左到右重排。
 */
export function canonicalizeLineStrokesForNeural(strokes: Stroke[]): Stroke[] {
  const valid = strokes.filter(s => s.tool !== 'eraser' && s.points.length >= 1);
  if (valid.length <= 2) return valid;

  const meta = valid.map((s, idx) => {
    const m = measureStroke(s);
    const pts = s.points;
    const dxNet = Math.abs(pts[pts.length - 1].x - pts[0].x);
    const dyNet = Math.abs(pts[pts.length - 1].y - pts[0].y);
    const horiz = dxNet / Math.max(1, dxNet + dyNet);
    const vert = dyNet / Math.max(1, dxNet + dyNet);
    return {
      idx,
      stroke: s,
      x0: m.bbox.x,
      y0: m.bbox.y,
      w: m.bbox.w,
      h: m.bbox.h,
      cx: m.cx,
      cy: m.cy,
      horiz,
      vert,
    };
  });

  const heights = meta.map(m => m.h).filter(h => h > 4 && h < 220).sort((a, b) => a - b);
  const avgH = Math.max(14, heights.length > 0 ? heights[Math.floor(heights.length / 2)] : 36);
  const used = new Set<number>();
  const orderedGroups: Array<{ cx: number; indices: number[] }> = [];

  for (const m of meta) {
    if (m.horiz > 0.82 && m.w >= avgH * 0.85 && m.h <= Math.max(18, m.w * 0.35)) {
      const nums: typeof meta = [];
      const dens: typeof meta = [];
      const bx0 = m.x0;
      const bx1 = m.x0 + m.w;
      const bcy = m.cy;
      for (const o of meta) {
        if (o.idx === m.idx || used.has(o.idx)) continue;
        const ox0 = o.x0;
        const ox1 = o.x0 + o.w;
        const inter = Math.max(0, Math.min(bx1, ox1) - Math.max(bx0, ox0));
        const xOv = inter / Math.max(1, Math.min(m.w, o.w));
        if (xOv < 0.2) continue;
        if (bcy - o.cy >= 3 && bcy - o.cy <= avgH * 1.75) {
          nums.push(o);
        } else if (o.cy - bcy >= 3 && o.cy - bcy <= avgH * 1.75) {
          dens.push(o);
        }
      }
      if (nums.length > 0 && dens.length > 0) {
        used.add(m.idx);
        nums.forEach(o => used.add(o.idx));
        dens.forEach(o => used.add(o.idx));
        nums.sort((a, b) => a.cx - b.cx || a.idx - b.idx);
        dens.sort((a, b) => a.cx - b.cx || a.idx - b.idx);
        orderedGroups.push({
          cx: m.cx,
          indices: [...nums.map(k => k.idx), m.idx, ...dens.map(k => k.idx)],
        });
      }
    }
  }

  for (const m of meta) {
    if (used.has(m.idx)) continue;
    if (m.h >= avgH * 1.3 && m.vert > 0.55 && m.w <= m.h * 0.75) {
      const lows: typeof meta = [];
      const highs: typeof meta = [];
      for (const o of meta) {
        if (o.idx === m.idx || used.has(o.idx)) continue;
        if (o.h < avgH * 0.95 && Math.abs(o.cx - (m.cx + m.w * 0.35)) <= avgH * 1.2) {
          if (o.cy >= m.cy + m.h * 0.2) lows.push(o);
          else if (o.cy <= m.cy - m.h * 0.2) highs.push(o);
        }
      }
      if (lows.length > 0 || highs.length > 0) {
        used.add(m.idx);
        lows.forEach(o => used.add(o.idx));
        highs.forEach(o => used.add(o.idx));
        lows.sort((a, b) => a.cx - b.cx || a.idx - b.idx);
        highs.sort((a, b) => a.cx - b.cx || a.idx - b.idx);
        orderedGroups.push({
          cx: m.cx,
          indices: [m.idx, ...lows.map(k => k.idx), ...highs.map(k => k.idx)],
        });
      }
    }
  }

  for (const m of meta) {
    if (!used.has(m.idx)) {
      orderedGroups.push({ cx: m.cx, indices: [m.idx] });
    }
  }

  orderedGroups.sort((a, b) => a.cx - b.cx || a.indices[0] - b.indices[0]);
  const reordered: Stroke[] = [];
  for (const grp of orderedGroups) {
    for (const idx of grp.indices) {
      reordered.push(valid[idx]);
    }
  }
  return reordered;
}

/**
 * 端侧全大学数学手写语法编码与图谱概率对齐引擎（实现 IInkSyntaxEncoderAdapter）。
 *
 * 【自研实现】设计思路：
 * 1. Quote ①：空间拓扑重排 `canonicalizeLineStrokesForNeural` 消除补写时间乱序；
 * 2. Quote ②③：10 通道 `EdgeMathGraphNet` + `StrokeGraphEdgeHead` 辅助多笔聚簇/二维关系判别 + TAMER 语法栈束搜索 & 题谱先验浅融合；
 * 3. 图二/图四：`evaluateGraphLogitHeads` 用每题动态 `stepLogits` / `pitfallLogits` 连续计算 `alignmentScore`、`pitfallProb`、`anchorGap` 与 `stepProgress`。
 */
export class EdgeInkSyntaxEncoder implements IInkSyntaxEncoderAdapter {
  private neuralAdapter: IInkSyntaxEncoderAdapter | null = null;
  private neuralRunner: EdgeNeuralModelRunner | null = null;
  private strokeSemCache = new Map<number, { ptCount: number; emb: Float32Array }>();

  public setNeuralEncoder(adapter: IInkSyntaxEncoderAdapter | null): void {
    this.neuralAdapter = adapter;
  }

  public setNeuralRunner(runner: EdgeNeuralModelRunner | null): void {
    this.neuralRunner = runner;
    this.strokeSemCache.clear();
  }

  private getRunner(): EdgeNeuralModelRunner {
    return this.neuralRunner ?? getSharedNeuralRunner();
  }

  private getCachedStrokeSem(stroke: Stroke, runner: EdgeNeuralModelRunner): Float32Array {
    const hit = this.strokeSemCache.get(stroke.id);
    if (hit && hit.ptCount === stroke.points.length) {
      return hit.emb;
    }
    const emb = runner.inferSingleStrokeSemantic(stroke);
    if (this.strokeSemCache.size >= 180) {
      this.strokeSemCache.clear();
    }
    this.strokeSemCache.set(stroke.id, { ptCount: stroke.points.length, emb });
    return emb;
  }

  public encodeSingleStrokeSemantic(stroke: Stroke): Float32Array {
    return this.getCachedStrokeSem(stroke, this.getRunner());
  }

  public isLoaded(): boolean {
    if (this.neuralAdapter) return this.neuralAdapter.isLoaded();
    return this.getRunner().isLoaded() || true;
  }

  public async encodeRecentStrokes(strokes: Stroke[]): Promise<{
    recognizedLatex: string;
    alignmentScore: number;
  }> {
    const detailed = await this.encodeWithSolution(strokes, null);
    return {
      recognizedLatex: detailed.recognizedLatex,
      alignmentScore: detailed.alignmentScore,
    };
  }

  private buildDecodeOptionsForSolution(solution: SolutionCacheEntry | null): CtcDecodeOptions {
    if (!solution) {
      return { enforceGrammar: true };
    }
    const expectedTokens = solution.steps.flatMap(s => s.expectedTokens ?? []);
    const stemTokens =
      solution.problemText.match(
        /\\(?:frac|sqrt|int|oint|sum|prod|partial|nabla|lim|to|det|matrix|lambda|alpha|beta|Phi)|\|A\|/g,
      ) ?? [];
    const priorLogitBias = buildProblemVocabPriorBias(expectedTokens, stemTokens);
    return { priorLogitBias, enforceGrammar: true };
  }

  public async encodeWithSolution(
    rawStrokes: Stroke[],
    solution: SolutionCacheEntry | null,
  ): Promise<EncodedExpressionResult> {
    const visible = filterVisiblePenStrokes(rawStrokes);
    const runner = this.getRunner();
    if (visible.length === 0) {
      return {
        recognizedLatex: '',
        structuralRole: 'scratch_draft',
        detectedDomain: 'general',
        structuralBoundaryFactor: 1.0,
        alignmentScore: 0,
        matchedStepIndex: -1,
        hitPitfall: null,
        pitfallProb: 0,
        anchorGap: 0,
        stepProgress: 0,
        activeLineCount: 0,
        tokenCount: 0,
        encoderMode: runner.isLoaded() ? 'neural_quantized_ts' : 'geometric_heuristic',
      };
    }

    const parsed = this.segmentAndClassify(visible, runner);
    const targetStrokes =
      parsed.lineCount >= 2 && !parsed.hasMatrixLayout && parsed.activeLineStrokes.length > 0
        ? parsed.activeLineStrokes
        : visible;
    const recentWindow = targetStrokes.length > 32 ? targetStrokes.slice(-32) : targetStrokes;
    const canonicalWindow = canonicalizeLineStrokesForNeural(recentWindow);
    const decodeOptions = this.buildDecodeOptionsForSolution(solution);

    let recognizedLatex = parsed.latex;
    let encoderMode: EncodedExpressionResult['encoderMode'] = 'geometric_heuristic';
    let tokenPosteriors: Float32Array | undefined;
    let externalAlignScore = 0;
    let hasExternalAlign = false;

    if (this.neuralAdapter?.isLoaded()) {
      try {
        if (this.neuralAdapter.encodeRecentStrokesDetailed) {
          const neural = await this.neuralAdapter.encodeRecentStrokesDetailed(
            canonicalWindow,
            decodeOptions,
          );
          if (neural.recognizedLatex.trim().length > 0) {
            recognizedLatex = neural.recognizedLatex.trim();
          }
          tokenPosteriors = neural.tokenPosteriors;
          externalAlignScore = neural.alignmentScore;
          hasExternalAlign = true;
        } else {
          const neural = await this.neuralAdapter.encodeRecentStrokes(canonicalWindow);
          if (neural.recognizedLatex.trim().length > 0) {
            recognizedLatex = neural.recognizedLatex.trim();
          }
          externalAlignScore = neural.alignmentScore;
          hasExternalAlign = true;
        }
        encoderMode = 'neural_onnx';
      } catch {
        // 外部 ONNX 适配器不可用时回退至内置量化模型或几何解码器
      }
    }

    if (encoderMode === 'geometric_heuristic' && runner.isLoaded()) {
      const neuralOut = runner.inferMathStrokes(canonicalWindow, 192, false, 'beam', decodeOptions);
      if (neuralOut.recognizedLatex.trim().length > 0) {
        recognizedLatex = neuralOut.recognizedLatex.trim();
      }
      tokenPosteriors = neuralOut.tokenPosteriors;
      encoderMode = 'neural_quantized_ts';
    }

    const aligned = this.evaluateGraphLogitHeads(
      recognizedLatex,
      parsed,
      tokenPosteriors,
      solution,
    );
    const finalAlign = hasExternalAlign
      ? clamp((externalAlignScore + aligned.alignmentScore) / 2, -1, 1)
      : aligned.alignmentScore;

    return {
      recognizedLatex,
      structuralRole: parsed.latestRole,
      detectedDomain: parsed.detectedDomain,
      structuralBoundaryFactor: parsed.boundaryFactor,
      alignmentScore: Number(finalAlign.toFixed(2)),
      matchedStepIndex: aligned.matchedStepIndex,
      hitPitfall: aligned.hitPitfall,
      pitfallProb: aligned.pitfallProb,
      anchorGap: aligned.anchorGap,
      stepProgress: aligned.stepProgress,
      tokenPosteriors: aligned.obsPosteriors,
      activeLineCount: parsed.lineCount,
      tokenCount: parsed.tokens.length,
      encoderMode,
    };
  }

  private segmentAndClassify(
    visibleStrokes: Stroke[],
    runner?: EdgeNeuralModelRunner,
  ): {
    latex: string;
    tokens: StrokeSymbolToken[];
    lineCount: number;
    latestRole: StructuralRole;
    detectedDomain: MathDomainHint;
    hasMatrixLayout: boolean;
    boundaryFactor: number;
    activeLineStrokes: Stroke[];
    graphEdgeHits: { hasAboveBelow: boolean; hasSupSub: boolean; wideFracCount: number };
  } {
    const metrics = visibleStrokes.map(measureStroke);
    const normalHeights = metrics
      .map(m => m.bbox.h)
      .filter(h => h > 8 && h < 180)
      .sort((a, b) => a - b);
    const medianH =
      normalHeights.length > 0 ? normalHeights[Math.floor(normalHeights.length / 2)] : 36;
    const avgHeight = clamp(medianH, 18, 110);

    // 1. 识别跨行成对大型定界符（如矩阵括号或行列式长竖线，排除竖直拉长的 ∫ / 字母）
    const tallDelimiters = metrics.filter(
      m =>
        m.bbox.h >= avgHeight * 2.0 &&
        m.verticalness > 0.72 &&
        m.bbox.h / Math.max(1, m.bbox.w) >= 2.4 &&
        m.chordDeviation < 0.24,
    );
    const hasMatrixLayout = tallDelimiters.length >= 2;
    const regularMetrics = hasMatrixLayout
      ? metrics.filter(m => !tallDelimiters.includes(m))
      : metrics;

    // 2. 仅当长横线上下两侧均存在非等号分子/分母笔画时，才视作真实分式线并合入同一推导行
    const candidateBars = regularMetrics.filter(
      m => m.horizontalness > 0.84 && m.bbox.w >= avgHeight * 0.9 && m.chordDeviation < 0.14,
    );
    const wideFracBars = candidateBars.filter(bar => {
      const hasNum = regularMetrics.some(
        s =>
          s !== bar &&
          !(s.horizontalness > 0.84 && s.bbox.h <= 6) &&
          xOverlapRatio(bar.bbox, s.bbox) >= 0.2 &&
          bar.cy - s.cy >= 4 &&
          bar.cy - s.cy <= avgHeight * 1.6,
      );
      const hasDen = regularMetrics.some(
        s =>
          s !== bar &&
          !(s.horizontalness > 0.84 && s.bbox.h <= 6) &&
          xOverlapRatio(bar.bbox, s.bbox) >= 0.2 &&
          s.cy - bar.cy >= 4 &&
          s.cy - bar.cy <= avgHeight * 1.6,
      );
      return hasNum && hasDen;
    });

    // 3. 普通笔画按垂直位置聚合成公式推导行（仅在确认真实分式时保留上下界在同一行）
    const lines: StrokeMetrics[][] = [];
    for (const m of regularMetrics) {
      let targetLine: StrokeMetrics[] | null = null;
      for (const line of lines) {
        const lineBox = mergeBbox(line.map(x => x.bbox));
        const lineCenterY = lineBox.y + lineBox.h / 2;
        const nearFracBand = wideFracBars.some(
          bar =>
            (line.includes(bar) || bar === m) &&
            xOverlapRatio(bar.bbox, line.includes(bar) ? m.bbox : lineBox) >= 0.2 &&
            Math.abs((line.includes(bar) ? bar.cy : m.cy) - (line.includes(bar) ? m.cy : lineCenterY)) <=
              avgHeight * 1.65,
        );
        if (
          yOverlapRatio(lineBox, m.bbox) >= 0.18 ||
          Math.abs(lineCenterY - m.cy) <= avgHeight * 0.92 ||
          nearFracBand
        ) {
          targetLine = line;
          break;
        }
      }
      if (targetLine) targetLine.push(m);
      else lines.push([m]);
    }
    if (lines.length === 0 && tallDelimiters.length > 0) {
      lines.push([...tallDelimiters]);
    }

    lines.sort((a, b) => mergeBbox(a.map(x => x.bbox)).y - mergeBbox(b.map(x => x.bbox)).y);

    // 4. 行内符号聚类（融合 Quote ③ StrokeGraphEdgeHead 成对神经图关系边）+ 上下标/均值修饰
    const allTokens: StrokeSymbolToken[] = [];
    const lineStrings: string[] = [];
    let hasAboveBelow = wideFracBars.length > 0;
    let hasSupSub = false;
    const useEdgeHead = Boolean(runner && runner.hasEdgeHead());

    for (const line of lines) {
      const sorted = [...line].sort((a, b) => a.cx - b.cx);
      const sameSymbolPairs = new Set<string>();

      if (useEdgeHead && runner && sorted.length >= 2 && sorted.length <= 48) {
        const sems = sorted.map(item => this.getCachedStrokeSem(item.stroke, runner));
        for (let i = 0; i < sorted.length; i++) {
          const maxJ = Math.min(sorted.length - 1, i + 3);
          for (let j = i + 1; j <= maxJ; j++) {
            const pred = runner.inferStrokeGraphEdges(
              sems[i],
              sems[j],
              extractPairGeom10(sorted[i].stroke, sorted[j].stroke, avgHeight),
            );
            if (pred.relation === 'SAME_SYMBOL' && (pred.probs[1] ?? 0) >= 0.48) {
              sameSymbolPairs.add(`${sorted[i].stroke.id}:${sorted[j].stroke.id}`);
              sameSymbolPairs.add(`${sorted[j].stroke.id}:${sorted[i].stroke.id}`);
            } else if (pred.relation === 'ABOVE' || pred.relation === 'BELOW') {
              hasAboveBelow = true;
            } else if (pred.relation === 'SUP' || pred.relation === 'SUB') {
              hasSupSub = true;
            }
          }
        }
      }

      const clusters: StrokeMetrics[][] = [];
      for (const m of sorted) {
        let joined = false;
        for (let idx = clusters.length - 1; idx >= 0; idx--) {
          const cluster = clusters[idx];
          const cBox = mergeBbox(cluster.map(x => x.bbox));
          const gapX = Math.max(m.bbox.x - (cBox.x + cBox.w), cBox.x - (m.bbox.x + m.bbox.w));
          const overlapX = xOverlapRatio(cBox, m.bbox) > 0.25;
          const lastInCluster = cluster[cluster.length - 1];
          const timeNear =
            Math.abs(
              m.stroke.points[0].t - lastInCluster.stroke.points[lastInCluster.stroke.points.length - 1].t,
            ) <= 2200;
          const neuralSame =
            gapX <= avgHeight * 0.75 &&
            sameSymbolPairs.has(`${lastInCluster.stroke.id}:${m.stroke.id}`);
          if (neuralSame || ((overlapX || gapX <= Math.max(10, avgHeight * 0.22)) && timeNear)) {
            cluster.push(m);
            joined = true;
            break;
          }
        }
        if (!joined) clusters.push([m]);
      }

      const lineTokens = clusters.map(c => classifyCluster(c, avgHeight));
      const annotatedStrings: string[] = [];
      for (let i = 0; i < lineTokens.length; i++) {
        const cur = lineTokens[i];
        const prev = i > 0 ? lineTokens[i - 1] : null;
        if (
          prev &&
          cur.bbox.h < prev.bbox.h * 0.68 &&
          cur.bbox.y + cur.bbox.h * 0.5 < prev.bbox.y + prev.bbox.h * 0.35
        ) {
          if (cur.token === '-' && xOverlapRatio(cur.bbox, prev.bbox) > 0.4) {
            annotatedStrings[annotatedStrings.length - 1] = `\\bar{${prev.token}}`;
            continue;
          }
          hasSupSub = true;
          annotatedStrings.push(`^${cur.token}`);
          continue;
        }
        if (
          prev &&
          cur.bbox.h < prev.bbox.h * 0.68 &&
          cur.bbox.y > prev.bbox.y + prev.bbox.h * 0.55
        ) {
          hasSupSub = true;
          annotatedStrings.push(`_${cur.token}`);
          continue;
        }
        annotatedStrings.push(cur.token);
      }

      allTokens.push(...lineTokens);
      lineStrings.push(annotatedStrings.join(' '));
    }

    if (hasMatrixLayout) {
      allTokens.push({
        token: '\\matrix',
        confidence: 0.85,
        role: 'matrix_delimiter',
        domain: 'linear_algebra',
        bbox: mergeBbox(tallDelimiters.map(d => d.bbox)),
        strokeIds: tallDelimiters.map(d => d.stroke.id),
      });
    }

    const latestStrokeId = visibleStrokes[visibleStrokes.length - 1].id;
    const latestToken =
      [...allTokens].reverse().find(t => t.strokeIds.includes(latestStrokeId)) ??
      allTokens[allTokens.length - 1];

    let detectedDomain: MathDomainHint = 'general';
    if (allTokens.some(t => t.domain === 'linear_algebra') || hasMatrixLayout) {
      detectedDomain = 'linear_algebra';
    } else if (allTokens.some(t => t.domain === 'calculus')) {
      detectedDomain = 'calculus';
    } else if (allTokens.some(t => t.domain === 'probability')) {
      detectedDomain = 'probability';
    }

    const latestRole: StructuralRole = latestToken?.role ?? 'scratch_draft';
    const boundaryFactor =
      latestRole === 'equation_rel' ||
      latestRole === 'fraction_bar' ||
      latestRole === 'calculus_operator' ||
      latestRole === 'matrix_delimiter' ||
      latestRole === 'probability_rel'
        ? 1.42
        : latestRole === 'logical_connector'
          ? 1.26
          : latestRole === 'operand'
            ? 0.8
            : 0.78;

    const activeLine =
      lines.find(line => line.some(m => m.stroke.id === latestStrokeId)) ??
      lines[lines.length - 1] ??
      [];
    const activeLineStrokes = activeLine.map(m => m.stroke);

    const previewPrefix = hasMatrixLayout ? '[矩阵/行列式] ' : '';
    return {
      latex: previewPrefix + lineStrings.slice(-2).join(' ; '),
      tokens: allTokens,
      lineCount: lines.length,
      latestRole,
      detectedDomain,
      hasMatrixLayout,
      boundaryFactor,
      activeLineStrokes,
      graphEdgeHits: {
        hasAboveBelow,
        hasSupSub,
        wideFracCount: wideFracBars.length,
      },
    };
  }

  /**
   * 图二/图四核心：连续可微解题图谱 Logit 头评测（彻底替换固定 4 条易错硬编码分支）。
   * 联合融合 CTC 边缘后验 P_ctc(v) 与二维几何观测后验 P_geo(v)，对每道题动态生成的
   * `stepLogits` 与 `pitfallLogits` 执行概率内积，输出：
   * `alignmentScore`、`matchedStepIndex`、`hitPitfall`、`pitfallProb`、`anchorGap` 与 `stepProgress`。
   */
  private evaluateGraphLogitHeads(
    latex: string,
    parsed: {
      tokens: StrokeSymbolToken[];
      lineCount: number;
      hasMatrixLayout: boolean;
      graphEdgeHits: { hasAboveBelow: boolean; hasSupSub: boolean; wideFracCount: number };
    },
    neuralPosteriors: Float32Array | undefined,
    solution: SolutionCacheEntry | null,
  ): {
    alignmentScore: number;
    matchedStepIndex: number;
    hitPitfall: SolutionPitfallSpec | null;
    pitfallProb: number;
    anchorGap: number;
    stepProgress: number;
    obsPosteriors: Float32Array;
  } {
    const obs = new Float32Array(MATH_VOCAB_SIZE);
    if (neuralPosteriors) {
      const nCopy = Math.min(obs.length, neuralPosteriors.length);
      for (let i = 0; i < nCopy; i++) {
        obs[i] = clamp(neuralPosteriors[i], 0, 1);
      }
    }

    const boostTok = (tok: string, val: number) => {
      const tid = MATH_VOCAB_INDEX.get(tok);
      if (tid !== undefined && tid > 0 && val > obs[tid]) {
        obs[tid] = clamp(val, 0, 1);
      }
    };

    for (const t of parsed.tokens) {
      boostTok(t.token, t.confidence);
      if (t.token === '/') boostTok('\\frac', t.confidence * 0.75);
      if (t.token === '\\matrix') boostTok('|', t.confidence * 0.85);
    }

    // 融合识别得到的 LaTeX 符号串中的显式线索
    const latexCmds = latex.match(/\\[a-zA-Z]+|[0-9a-zA-Z=+\-*/^_|(){}\[\],.<>!]/g) ?? [];
    for (const tok of latexCmds) {
      boostTok(tok, 0.82);
    }
    if (parsed.graphEdgeHits.wideFracCount > 0 || parsed.graphEdgeHits.hasAboveBelow) {
      boostTok('\\frac', 0.82);
    }
    if (parsed.graphEdgeHits.hasSupSub) {
      boostTok('^', 0.72);
      boostTok('_', 0.72);
    }

    const roleAct: Record<GraphGeometricRole, number> = {
      logical_connector: 0,
      fraction_bar: 0,
      equation_rel: 0,
      calculus_operator: 0,
      matrix_delimiter: 0,
      probability_rel: 0,
      operand: 0,
    };
    for (const t of parsed.tokens) {
      if (t.role !== 'scratch_draft') {
        roleAct[t.role] = Math.max(roleAct[t.role], t.confidence);
      }
    }
    if (parsed.hasMatrixLayout) roleAct.matrix_delimiter = Math.max(roleAct.matrix_delimiter, 0.9);
    if (parsed.graphEdgeHits.wideFracCount > 0 || (obs[MATH_VOCAB_INDEX.get('\\frac') ?? -1] ?? 0) >= 0.55) {
      roleAct.fraction_bar = Math.max(roleAct.fraction_bar, 0.85);
    }
    if ((obs[MATH_VOCAB_INDEX.get('=') ?? -1] ?? 0) >= 0.5 || latex.includes('=')) {
      roleAct.equation_rel = Math.max(roleAct.equation_rel, 0.85);
    }
    if (
      (obs[MATH_VOCAB_INDEX.get('\\int') ?? -1] ?? 0) >= 0.5 ||
      (obs[MATH_VOCAB_INDEX.get('\\lim') ?? -1] ?? 0) >= 0.5
    ) {
      roleAct.calculus_operator = Math.max(roleAct.calculus_operator, 0.85);
    }

    const hasAnyOps =
      roleAct.equation_rel >= 0.45 ||
      roleAct.fraction_bar >= 0.45 ||
      roleAct.calculus_operator >= 0.45 ||
      roleAct.matrix_delimiter >= 0.45 ||
      roleAct.logical_connector >= 0.45;

    if (!solution || parsed.tokens.length === 0) {
      return {
        alignmentScore: hasAnyOps ? 0.28 : 0,
        matchedStepIndex: -1,
        hitPitfall: null,
        pitfallProb: 0,
        anchorGap: 0,
        stepProgress: 0,
        obsPosteriors: obs,
      };
    }

    // 1. 评测动态生成的每题 PitfallLogitHeadSpec（连续概率逻辑回归，支持题谱泛化）
    let maxPitfallProb = 0;
    let hitPitfall: SolutionPitfallSpec | null = null;
    let bestPitfallProbTriggered = 0;

    for (const pitfall of solution.commonPitfalls) {
      const head = pitfall.pitfallLogits;
      if (!head) continue;
      let z = head.bias + head.lineCountWeight * parsed.lineCount;
      for (const [tok, w] of Object.entries(head.presentWeights)) {
        const tid = MATH_VOCAB_INDEX.get(tok);
        const pSeen = tid !== undefined && tid > 0 ? obs[tid] : 0;
        z += w * pSeen;
      }
      for (const [tok, w] of Object.entries(head.missingWeights)) {
        const tid = MATH_VOCAB_INDEX.get(tok);
        const pSeen = tid !== undefined && tid > 0 ? obs[tid] : 0;
        z += w * (1 - pSeen);
      }
      for (const [role, w] of Object.entries(head.roleWeights)) {
        const rAct = roleAct[role as GraphGeometricRole] ?? 0;
        z += (w ?? 0) * rAct;
      }
      const prob = sigmoid(z);
      if (prob > maxPitfallProb) {
        maxPitfallProb = prob;
      }
      if (
        parsed.lineCount >= head.minLineCount &&
        prob >= head.threshold &&
        prob > bestPitfallProbTriggered
      ) {
        bestPitfallProbTriggered = prob;
        hitPitfall = pitfall;
      }
    }

    // 2. 评测动态生成的 StepLogitHeadSpec（计算每步覆盖率、anchorGap 与 stepProgress）
    if (solution.steps.length === 0) {
      const fallbackAlign = hitPitfall
        ? -clamp(0.45 + 0.35 * maxPitfallProb, 0.55, 0.85)
        : clamp((hasAnyOps ? 0.28 : 0.05) - 0.35 * Math.max(0, maxPitfallProb - 0.35), -0.8, 0.9);
      return {
        alignmentScore: Number(fallbackAlign.toFixed(2)),
        matchedStepIndex: 0,
        hitPitfall,
        pitfallProb: Number(maxPitfallProb.toFixed(3)),
        anchorGap: 0,
        stepProgress: parsed.lineCount > 0 ? 0.5 : 0,
        obsPosteriors: obs,
      };
    }

    const totalSteps = solution.steps.length;
    const empProgress = clamp(parsed.lineCount / Math.max(1, totalSteps), 0, 1);
    let bestStepIdx = 0;
    let bestStepFit = -Infinity;
    let bestStepCoverage = 0;
    let bestAnchorGap = 0;

    for (let i = 0; i < totalSteps; i++) {
      const step = solution.steps[i];
      const head = step.stepLogits;
      let weightedHit = 0;
      let totalWeight = 0;
      let missingMass = 0;

      if (head) {
        for (const [tok, wRaw] of Object.entries(head.tokenWeights)) {
          const w = Math.max(0, wRaw);
          if (w <= 0) continue;
          const tid = MATH_VOCAB_INDEX.get(tok);
          const pSeen = tid !== undefined && tid > 0 ? obs[tid] : 0;
          weightedHit += w * pSeen;
          missingMass += w * (1 - pSeen);
          totalWeight += w;
        }
        for (const [role, wRaw] of Object.entries(head.roleWeights)) {
          const w = Math.max(0, wRaw ?? 0);
          if (w <= 0) continue;
          const rAct = roleAct[role as GraphGeometricRole] ?? 0;
          weightedHit += w * rAct;
          missingMass += w * (1 - rAct);
          totalWeight += w;
        }
      }

      const denom = Math.max(0.25, totalWeight);
      const coverage = clamp(weightedHit / denom, 0, 1);
      const gap = clamp(missingMass / denom, 0, 1);
      const progressCenter = head?.progressCenter ?? (totalSteps > 1 ? i / (totalSteps - 1) : 0.5);
      const fitScore = coverage * 1.15 - Math.abs(empProgress - progressCenter) * 0.25;

      if (fitScore > bestStepFit) {
        bestStepFit = fitScore;
        bestStepIdx = i;
        bestStepCoverage = coverage;
        bestAnchorGap = gap;
      }
    }

    const stepProgress = clamp((bestStepIdx + bestStepCoverage) / Math.max(1, totalSteps), 0, 1);
    const pitfallPenalty = hitPitfall
      ? clamp(0.45 + 0.35 * maxPitfallProb, 0.55, 0.85)
      : Math.max(0, maxPitfallProb - 0.38) * 0.65;
    const rawAlign = hitPitfall
      ? -pitfallPenalty
      : 0.14 + bestStepFit * 0.68 + (hasAnyOps ? 0.12 : 0) - pitfallPenalty;

    return {
      alignmentScore: Number(clamp(rawAlign, -0.85, 0.92).toFixed(2)),
      matchedStepIndex: bestStepIdx,
      hitPitfall,
      pitfallProb: Number(maxPitfallProb.toFixed(3)),
      anchorGap: Number(bestAnchorGap.toFixed(3)),
      stepProgress: Number(stepProgress.toFixed(3)),
      obsPosteriors: obs,
    };
  }
}
