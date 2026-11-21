import type { Point, Stroke } from '../capture';
import { REWRITE_RADIUS } from '../stalls/config';
import { EdgeNeuralModelRunner } from './neuralRuntime';
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

/**
 * 端侧全大学数学手写语法编码与步骤对齐引擎（实现 IInkSyntaxEncoderAdapter）。
 *
 * 【自研实现】设计思路：
 * 1. 覆盖高等数学（\int, \oint, \sum, \prod, \partial, \nabla, \infty, \to, \lim）、
 *    线性代数（矩阵/行列式跨行界定符与二维块、转置/逆矩阵上标 ^T/^{-1}、下标 _{ij}）
 *    与概率统计（~, \Phi, \cap, \cup, \varnothing, \bar{X} 上划线, P(A|B)）常见符号及二维布局；
 * 2. 在逻辑算符与边界锚点（=, \Rightarrow, \int, \sum, \to, \frac, |matrix|）后出现停顿时放大结构惊奇度权重，
 *    而在普通操作数（数字/变量）书写中途降低惊奇权重，避免干扰深度心算；
 * 3. 与解题图谱（SolutionCacheEntry）执行步骤对齐（A(t)）与易错歧路扫描（如阶数不够、漏讨论 |A|=0、条件概率分子分母颠倒）。
 */
export class EdgeInkSyntaxEncoder implements IInkSyntaxEncoderAdapter {
  private neuralAdapter: IInkSyntaxEncoderAdapter | null = null;
  private neuralRunner: EdgeNeuralModelRunner | null = null;

  public setNeuralEncoder(adapter: IInkSyntaxEncoderAdapter | null): void {
    this.neuralAdapter = adapter;
  }

  public setNeuralRunner(runner: EdgeNeuralModelRunner | null): void {
    this.neuralRunner = runner;
  }

  private getRunner(): EdgeNeuralModelRunner {
    return this.neuralRunner ?? getSharedNeuralRunner();
  }

  public encodeSingleStrokeSemantic(stroke: Stroke): Float32Array {
    return this.getRunner().inferSingleStrokeSemantic(stroke);
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

  public async encodeWithSolution(
    rawStrokes: Stroke[],
    solution: SolutionCacheEntry | null,
  ): Promise<EncodedExpressionResult> {
    const visible = filterVisiblePenStrokes(rawStrokes);
    if (visible.length === 0) {
      return {
        recognizedLatex: '',
        structuralRole: 'scratch_draft',
        detectedDomain: 'general',
        structuralBoundaryFactor: 1.0,
        alignmentScore: 0,
        matchedStepIndex: -1,
        hitPitfall: null,
        activeLineCount: 0,
        tokenCount: 0,
        encoderMode: this.getRunner().isLoaded() ? 'neural_quantized_ts' : 'geometric_heuristic',
      };
    }

    if (this.neuralAdapter?.isLoaded()) {
      try {
        const neural = await this.neuralAdapter.encodeRecentStrokes(visible);
        const tokens = this.segmentAndClassify(visible);
        const aligned = this.alignTokensWithSolution(
          neural.recognizedLatex,
          tokens.tokens,
          tokens.lineCount,
          tokens.hasMatrixLayout,
          solution,
        );
        return {
          recognizedLatex: neural.recognizedLatex,
          structuralRole: tokens.latestRole,
          detectedDomain: tokens.detectedDomain,
          structuralBoundaryFactor: tokens.boundaryFactor,
          alignmentScore: clamp((neural.alignmentScore + aligned.alignmentScore) / 2, -1, 1),
          matchedStepIndex: aligned.matchedStepIndex,
          hitPitfall: aligned.hitPitfall,
          activeLineCount: tokens.lineCount,
          tokenCount: tokens.tokens.length,
          encoderMode: 'neural_onnx',
        };
      } catch {
        // 外部 ONNX 适配器不可用时回退至内置量化模型或几何解码器
      }
    }

    const parsed = this.segmentAndClassify(visible);
    const runner = this.getRunner();
    let recognizedLatex = parsed.latex;
    let encoderMode: EncodedExpressionResult['encoderMode'] = 'geometric_heuristic';

    if (runner.isLoaded()) {
      const targetStrokes =
        parsed.lineCount >= 2 && !parsed.hasMatrixLayout && parsed.activeLineStrokes.length > 0
          ? parsed.activeLineStrokes
          : visible;
      const recentWindow = targetStrokes.length > 32 ? targetStrokes.slice(-32) : targetStrokes;
      const neuralOut = runner.inferMathStrokes(recentWindow, 192);
      if (neuralOut.recognizedLatex.trim().length > 0) {
        recognizedLatex = neuralOut.recognizedLatex.trim();
      }
      encoderMode = 'neural_quantized_ts';
    }

    const aligned = this.alignTokensWithSolution(
      recognizedLatex,
      parsed.tokens,
      parsed.lineCount,
      parsed.hasMatrixLayout,
      solution,
    );

    return {
      recognizedLatex,
      structuralRole: parsed.latestRole,
      detectedDomain: parsed.detectedDomain,
      structuralBoundaryFactor: parsed.boundaryFactor,
      alignmentScore: aligned.alignmentScore,
      matchedStepIndex: aligned.matchedStepIndex,
      hitPitfall: aligned.hitPitfall,
      activeLineCount: parsed.lineCount,
      tokenCount: parsed.tokens.length,
      encoderMode,
    };
  }

  private segmentAndClassify(visibleStrokes: Stroke[]): {
    latex: string;
    tokens: StrokeSymbolToken[];
    lineCount: number;
    latestRole: StructuralRole;
    detectedDomain: MathDomainHint;
    hasMatrixLayout: boolean;
    boundaryFactor: number;
    activeLineStrokes: Stroke[];
  } {
    const metrics = visibleStrokes.map(measureStroke);
    const normalHeights = metrics
      .map(m => m.bbox.h)
      .filter(h => h > 6 && h < 55);
    const avgHeight =
      normalHeights.length > 0
        ? clamp(normalHeights.reduce((s, h) => s + h, 0) / normalHeights.length, 16, 38)
        : 28;

    // 1. 识别跨行成对大型定界符（如矩阵括号或行列式长竖线），并在行聚类时避免将其碎片化
    const tallDelimiters = metrics.filter(
      m => m.bbox.h >= avgHeight * 2.0 && m.verticalness > 0.6,
    );
    const hasMatrixLayout = tallDelimiters.length >= 2;
    const regularMetrics = hasMatrixLayout
      ? metrics.filter(m => !tallDelimiters.includes(m))
      : metrics;

    // 2. 普通笔画按垂直位置聚合成公式推导行
    const lines: StrokeMetrics[][] = [];
    for (const m of regularMetrics) {
      let targetLine: StrokeMetrics[] | null = null;
      for (const line of lines) {
        const lineBox = mergeBbox(line.map(x => x.bbox));
        const lineCenterY = lineBox.y + lineBox.h / 2;
        if (
          yOverlapRatio(lineBox, m.bbox) >= 0.2 ||
          Math.abs(lineCenterY - m.cy) <= avgHeight * 0.85
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

    // 3. 行内符号聚类 + 上下标/均值上划线二维关系修饰
    const allTokens: StrokeSymbolToken[] = [];
    const lineStrings: string[] = [];

    for (const line of lines) {
      const sorted = [...line].sort((a, b) => a.cx - b.cx);
      const clusters: StrokeMetrics[][] = [];
      for (const m of sorted) {
        let joined = false;
        for (let idx = clusters.length - 1; idx >= 0; idx--) {
          const cluster = clusters[idx];
          const cBox = mergeBbox(cluster.map(x => x.bbox));
          const gapX = Math.max(m.bbox.x - (cBox.x + cBox.w), cBox.x - (m.bbox.x + m.bbox.w));
          const overlapX = xOverlapRatio(cBox, m.bbox) > 0.25;
          const timeNear =
            Math.abs(
              m.stroke.points[0].t -
                cluster[cluster.length - 1].stroke.points[
                  cluster[cluster.length - 1].stroke.points.length - 1
                ].t,
            ) <= 2200;
          if ((overlapX || gapX <= Math.max(10, avgHeight * 0.22)) && timeNear) {
            cluster.push(m);
            joined = true;
            break;
          }
        }
        if (!joined) clusters.push([m]);
      }

      const lineTokens = clusters.map(c => classifyCluster(c, avgHeight));
      // 二维关系修饰：\bar{x} 上划线（均值）、上标 ^T / ^{-1}、下标 _i / _{ij}
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
          annotatedStrings.push(`^${cur.token}`);
          continue;
        }
        if (
          prev &&
          cur.bbox.h < prev.bbox.h * 0.68 &&
          cur.bbox.y > prev.bbox.y + prev.bbox.h * 0.55
        ) {
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

    // 推断主学科上下文
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
    };
  }

  private alignTokensWithSolution(
    latex: string,
    tokens: StrokeSymbolToken[],
    lineCount: number,
    hasMatrixLayout: boolean,
    solution: SolutionCacheEntry | null,
  ): {
    alignmentScore: number;
    matchedStepIndex: number;
    hitPitfall: SolutionPitfallSpec | null;
  } {
    if (!solution || tokens.length === 0) {
      return { alignmentScore: 0, matchedStepIndex: -1, hitPitfall: null };
    }

    const eqCount = Math.max(
      tokens.filter(t => t.token === '=' || t.token === '\\equiv').length,
      (latex.match(/=/g) ?? []).length,
    );
    const fracCount = Math.max(
      tokens.filter(t => t.token === '\\frac').length,
      latex.includes('\\frac') || latex.includes('/') ? 1 : 0,
    );
    const hasIntegral =
      tokens.some(t => t.token === '\\int' || t.token === '\\oint') ||
      latex.includes('\\int') ||
      latex.includes('\\oint');
    const hasSum = tokens.some(t => t.token === '\\sum') || latex.includes('\\sum');
    const hasLimitArrow =
      tokens.some(t => t.token === '\\to') || latex.includes('\\to') || latex.includes('\\lim');
    const hasProbSymbol =
      tokens.some(t => t.domain === 'probability') ||
      /P\(|Phi|\\cap|\\cup|\\sim/.test(latex);
    const hasZero = tokens.some(t => t.token === '0') || /(^|[^0-9])0([^0-9]|$)/.test(latex);
    const hasOps =
      tokens.some(
        t =>
          t.role === 'logical_connector' ||
          t.role === 'equation_rel' ||
          t.role === 'calculus_operator' ||
          t.role === 'matrix_delimiter',
      ) || eqCount > 0 || fracCount > 0 || hasIntegral || hasSum || hasLimitArrow;

    // 1. 扫描高数/线代/概率论共性易错歧路特征
    let hitPitfall: SolutionPitfallSpec | null = null;
    for (const pitfall of solution.commonPitfalls) {
      const pLower = pitfall.triggerPattern.toLowerCase();
      // 线性代数：讨论含参矩阵解的结构前，如果未求 |A|=0（缺少矩阵/行列式界定符或 '= 0' 检测）便写入多行推导
      if (
        (pLower.includes('|a|=0') || pLower.includes('行列式') || pLower.includes('漏讨论零点')) &&
        lineCount >= 2 &&
        eqCount >= 1 &&
        !hasZero &&
        !hasMatrixLayout
      ) {
        hitPitfall = pitfall;
        break;
      }
      // 高数求极限/泰勒：写了分式或等号多行展开但无高阶项/未写分母展开
      if (
        (pLower.includes('阶数不够') || pLower.includes('截断') || pLower.includes('洛必达')) &&
        lineCount >= 3 &&
        fracCount >= 2
      ) {
        hitPitfall = pitfall;
        break;
      }
      // 概率论贝叶斯公式：全概率分母求和或分式表达缺失
      if (
        (pLower.includes('全概率') || pLower.includes('分母') || pLower.includes('先验权重')) &&
        lineCount >= 2 &&
        eqCount >= 1 &&
        fracCount === 0 &&
        !hasSum
      ) {
        hitPitfall = pitfall;
        break;
      }
      // 经典定积分/分部积分或常数遗漏检查
      if (
        (pLower.includes('除以2') || pLower.includes('分式')) &&
        lineCount >= 2 &&
        eqCount >= 1 &&
        fracCount === 0
      ) {
        hitPitfall = pitfall;
        break;
      }
    }

    if (hitPitfall) {
      return {
        alignmentScore: -0.65,
        matchedStepIndex: Math.min(Math.max(0, lineCount - 1), Math.max(0, solution.steps.length - 1)),
        hitPitfall,
      };
    }

    // 2. 步骤主干匹配
    if (solution.steps.length === 0) {
      return {
        alignmentScore: hasOps ? 0.28 : 0,
        matchedStepIndex: 0,
        hitPitfall: null,
      };
    }

    let bestIdx = 0;
    let bestScore = -1;
    for (let i = 0; i < solution.steps.length; i++) {
      const step = solution.steps[i];
      const expected = step.expectedTokens.join(' ') + ' ' + step.goalDescription;
      let matchWeight = 0;
      if (expected.includes('=') && eqCount > 0) matchWeight += 0.25;
      if ((expected.includes('/') || expected.includes('frac')) && fracCount > 0) matchWeight += 0.3;
      if ((expected.includes('int') || expected.includes('积分')) && hasIntegral) matchWeight += 0.35;
      if ((expected.includes('sum') || expected.includes('求和')) && hasSum) matchWeight += 0.3;
      if ((expected.includes('to') || expected.includes('lim')) && hasLimitArrow) matchWeight += 0.3;
      if ((expected.includes('matrix') || expected.includes('det') || expected.includes('|A|')) && hasMatrixLayout) {
        matchWeight += 0.35;
      }
      if ((expected.includes('P(') || expected.includes('Phi') || expected.includes('~')) && hasProbSymbol) {
        matchWeight += 0.3;
      }
      if (expected.includes('0') && hasZero) matchWeight += 0.15;
      if (latex.includes('expr') || latex.includes('sym')) matchWeight += 0.12;

      const lineProgressDiff = Math.abs(Math.min(lineCount - 1, solution.steps.length - 1) - i);
      const structuralFit = matchWeight - lineProgressDiff * 0.12;
      if (structuralFit > bestScore) {
        bestScore = structuralFit;
        bestIdx = i;
      }
    }

    const positiveA = clamp(0.16 + bestScore * 0.72 + (hasOps ? 0.15 : 0), -0.2, 0.9);
    return {
      alignmentScore: Number(positiveA.toFixed(2)),
      matchedStepIndex: bestIdx,
      hitPitfall: null,
    };
  }
}
