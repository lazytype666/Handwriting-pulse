import type { Point, Stroke } from '../capture';
import {
  EDGE_NEURAL_BUNDLE,
  type EdgeNeuralBundle,
  type QuantizedTensorSpec,
} from './edgeNeuralWeights';
import {
  computeCtcTokenPosteriors,
  decodeCtcBeamSearch,
  decodeCtcGreedy,
  EDGE_RELATIONS,
  MATH_VOCAB_INDEX,
  MATH_VOCAB_SIZE,
  type CtcDecodeOptions,
  type StrokeEdgeRelation,
} from './mathVocab';

export type CognitiveCategory =
  | 'FLUENT_DERIVATION'
  | 'DEEP_THINKING_HOLD'
  | 'CALC_MICRO_HESITATION'
  | 'STUCK_CALCULATION'
  | 'STUCK_CONCEPT_GAP'
  | 'DIRECTION_DRIFT_PITFALL';

export const COGNITIVE_CATEGORIES: readonly CognitiveCategory[] = [
  'FLUENT_DERIVATION',
  'DEEP_THINKING_HOLD',
  'CALC_MICRO_HESITATION',
  'STUCK_CALCULATION',
  'STUCK_CONCEPT_GAP',
  'DIRECTION_DRIFT_PITFALL',
];

export const COGNITIVE_CATEGORY_ZH: Record<CognitiveCategory, string> = {
  FLUENT_DERIVATION: '顺畅推导',
  DEEP_THINKING_HOLD: '深度心算',
  CALC_MICRO_HESITATION: '算式微卡',
  STUCK_CALCULATION: '算式受阻',
  STUCK_CONCEPT_GAP: '思路断链',
  DIRECTION_DRIFT_PITFALL: '歧路偏离',
};

export interface NeuralMathOutput {
  recognizedLatex: string;
  semPooled: Float32Array;
  /** 125 维 CTC 符号边缘后验概率 P_ctc(v) ∈ [0, 1] */
  tokenPosteriors: Float32Array;
}

export interface StrokeEdgePrediction {
  bestClass: number;
  relation: StrokeEdgeRelation;
  probs: number[];
}

export interface NeuralCognitiveOutput {
  classProbs: number[];
  predictedClass: number;
  category: CognitiveCategory;
  surpriseScore: number;
  momentumM: number;
  resistanceR: number;
  alignmentA: number;
}

const B64_ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const B64_LUT = new Int16Array(256).fill(-1);
for (let i = 0; i < B64_ALPHA.length; i++) {
  B64_LUT[B64_ALPHA.charCodeAt(i)] = i;
}

function decodeBase64ToBytes(b64: string): Uint8Array {
  const clean = b64.replace(/=+$/, '');
  const outLen = (clean.length * 3) >> 2;
  const out = new Uint8Array(outLen);
  let p = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const c0 = B64_LUT[clean.charCodeAt(i)];
    const c1 = B64_LUT[clean.charCodeAt(i + 1)];
    const c2 = i + 2 < clean.length ? B64_LUT[clean.charCodeAt(i + 2)] : 0;
    const c3 = i + 3 < clean.length ? B64_LUT[clean.charCodeAt(i + 3)] : 0;
    const trip = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;
    if (p < outLen) out[p++] = (trip >> 16) & 0xff;
    if (p < outLen) out[p++] = (trip >> 8) & 0xff;
    if (p < outLen) out[p++] = trip & 0xff;
  }
  return out;
}

function dequantizeTensor(spec: QuantizedTensorSpec): Float32Array {
  if (spec.dtype === 'f32') {
    return new Float32Array(spec.data);
  }
  const bytes = decodeBase64ToBytes(spec.b64);
  const out = new Float32Array(bytes.length);
  const s = spec.scale;
  for (let i = 0; i < bytes.length; i++) {
    const signed = bytes[i] >= 128 ? bytes[i] - 256 : bytes[i];
    out[i] = signed * s;
  }
  return out;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-clamp(x, -30, 30)));
}

function siluInPlace(x: Float32Array): void {
  for (let i = 0; i < x.length; i++) {
    x[i] = x[i] / (1 + Math.exp(-clamp(x[i], -30, 30)));
  }
}

function linear(
  x: Float32Array,
  rows: number,
  inDim: number,
  outDim: number,
  w: Float32Array,
  b?: Float32Array,
): Float32Array {
  const out = new Float32Array(rows * outDim);
  for (let r = 0; r < rows; r++) {
    const xOff = r * inDim;
    const oOff = r * outDim;
    for (let o = 0; o < outDim; o++) {
      let sum = b ? b[o] : 0;
      const wOff = o * inDim;
      for (let k = 0; k < inDim; k++) {
        sum += x[xOff + k] * w[wOff + k];
      }
      out[oOff + o] = sum;
    }
  }
  return out;
}

function layerNorm(
  x: Float32Array,
  rows: number,
  dim: number,
  gamma: Float32Array,
  beta: Float32Array,
  inPlace = false,
): Float32Array {
  const out = inPlace ? x : new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    const off = r * dim;
    let mean = 0;
    for (let d = 0; d < dim; d++) mean += x[off + d];
    mean /= dim;
    let varSum = 0;
    for (let d = 0; d < dim; d++) {
      const diff = x[off + d] - mean;
      varSum += diff * diff;
    }
    const invStd = 1 / Math.sqrt(varSum / dim + 1e-5);
    for (let d = 0; d < dim; d++) {
      out[off + d] = (x[off + d] - mean) * invStd * gamma[d] + beta[d];
    }
  }
  return out;
}

function conv1dStride2Pad2K5(
  x: Float32Array,
  seqLen: number,
  inC: number,
  outC: number,
  w: Float32Array,
  b: Float32Array,
): { out: Float32Array; outLen: number } {
  const outLen = Math.max(1, Math.floor((seqLen + 4 - 5) / 2) + 1);
  const out = new Float32Array(outLen * outC);
  for (let tOut = 0; tOut < outLen; tOut++) {
    const center = tOut * 2 - 2;
    const oBase = tOut * outC;
    for (let cOut = 0; cOut < outC; cOut++) {
      let sum = b[cOut];
      const wBase = cOut * inC * 5;
      for (let cIn = 0; cIn < inC; cIn++) {
        const wCh = wBase + cIn * 5;
        for (let k = 0; k < 5; k++) {
          const tIn = center + k;
          if (tIn >= 0 && tIn < seqLen) {
            sum += x[tIn * inC + cIn] * w[wCh + k];
          }
        }
      }
      out[oBase + cOut] = sum;
    }
  }
  return { out, outLen };
}

function depthwiseConv1dK5Pad2(
  x: Float32Array,
  seqLen: number,
  dim: number,
  w: Float32Array,
  b: Float32Array,
): Float32Array {
  const out = new Float32Array(seqLen * dim);
  for (let t = 0; t < seqLen; t++) {
    const tBase = t * dim;
    for (let d = 0; d < dim; d++) {
      let sum = b[d];
      const wBase = d * 5;
      for (let k = 0; k < 5; k++) {
        const tIn = t - 2 + k;
        if (tIn >= 0 && tIn < seqLen) {
          sum += x[tIn * dim + d] * w[wBase + k];
        }
      }
      out[tBase + d] = sum;
    }
  }
  return out;
}

function multiheadAttention(
  qSrc: Float32Array,
  kSrc: Float32Array,
  vSrc: Float32Array,
  qLen: number,
  kvLen: number,
  dim: number,
  nHeads: number,
  inProjW: Float32Array,
  inProjB: Float32Array,
  outProjW: Float32Array,
  outProjB: Float32Array,
): Float32Array {
  const headDim = dim / nHeads;
  const scale = 1 / Math.sqrt(headDim);

  const wQ = inProjW.subarray(0, dim * dim);
  const wK = inProjW.subarray(dim * dim, 2 * dim * dim);
  const wV = inProjW.subarray(2 * dim * dim, 3 * dim * dim);
  const bQ = inProjB.subarray(0, dim);
  const bK = inProjB.subarray(dim, 2 * dim);
  const bV = inProjB.subarray(2 * dim, 3 * dim);

  const qAll = linear(qSrc, qLen, dim, dim, wQ, bQ);
  const kAll = linear(kSrc, kvLen, dim, dim, wK, bK);
  const vAll = linear(vSrc, kvLen, dim, dim, wV, bV);

  const ctx = new Float32Array(qLen * dim);
  const scores = new Float32Array(kvLen);

  for (let h = 0; h < nHeads; h++) {
    const hOff = h * headDim;
    for (let qi = 0; qi < qLen; qi++) {
      const qOff = qi * dim + hOff;
      let maxScore = -Infinity;
      for (let ki = 0; ki < kvLen; ki++) {
        const kOff = ki * dim + hOff;
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += qAll[qOff + d] * kAll[kOff + d];
        }
        const s = dot * scale;
        scores[ki] = s;
        if (s > maxScore) maxScore = s;
      }

      let expSum = 0;
      for (let ki = 0; ki < kvLen; ki++) {
        const e = Math.exp(scores[ki] - maxScore);
        scores[ki] = e;
        expSum += e;
      }
      const invSum = 1 / Math.max(1e-8, expSum);

      const outOff = qi * dim + hOff;
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let ki = 0; ki < kvLen; ki++) {
          acc += scores[ki] * invSum * vAll[ki * dim + hOff + d];
        }
        ctx[outOff + d] = acc;
      }
    }
  }

  return linear(ctx, qLen, dim, dim, outProjW, outProjB);
}

function gruForward(
  x: Float32Array,
  steps: number,
  dim: number,
  wIh: Float32Array,
  wHh: Float32Array,
  bIh: Float32Array,
  bHh: Float32Array,
): Float32Array {
  const outSeq = new Float32Array(steps * dim);
  const h = new Float32Array(dim);
  const gateDim = 3 * dim;

  for (let t = 0; t < steps; t++) {
    const xT = x.subarray(t * dim, (t + 1) * dim);
    const gi = linear(xT, 1, dim, gateDim, wIh, bIh);
    const gh = linear(h, 1, dim, gateDim, wHh, bHh);

    for (let d = 0; d < dim; d++) {
      const r = sigmoid(gi[d] + gh[d]);
      const z = sigmoid(gi[dim + d] + gh[dim + d]);
      const n = Math.tanh(gi[2 * dim + d] + r * gh[2 * dim + d]);
      const hNext = (1 - z) * n + z * h[d];
      h[d] = hNext;
      outSeq[t * dim + d] = hNext;
    }
  }
  return outSeq;
}

const DELIM_VOCAB_TOKENS = new Set(['(', ')', '[', ']', '{', '}', '_', '^', ',', '.']);

function tokenizeExprIntoIds(expr: string): number[] {
  const ids: number[] = [];
  let i = 0;
  while (i < expr.length) {
    const ch = expr[i];
    if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '$') {
      i += 1;
      continue;
    }
    if (ch === '\\') {
      let j = i + 1;
      while (j < expr.length && /[a-zA-Z]/.test(expr[j])) {
        j += 1;
      }
      const cmd = expr.slice(i, j);
      const id = MATH_VOCAB_INDEX.get(cmd);
      if (id !== undefined && id > 0) {
        ids.push(id);
      }
      i = Math.max(j, i + 1);
      continue;
    }
    const id = MATH_VOCAB_INDEX.get(ch);
    if (id !== undefined && id > 0) {
      ids.push(id);
    }
    i += 1;
  }
  return ids;
}

/**
 * 构建解题图谱浅融合对数先验偏置向量（长度 125）。
 * 根据当前步骤期望符号与题干数学式提高候选符号的 Beam Search 对数概率。
 */
export function buildProblemVocabPriorBias(
  expectedTokens: string[],
  stemLatexSegments: string[] = [],
): Float32Array | undefined {
  const bias = new Float32Array(MATH_VOCAB_SIZE);
  let hasAny = false;

  const applyTokens = (sources: string[], baseWeight: number, maxWeight: number) => {
    for (const src of sources) {
      if (!src) continue;
      const directId = MATH_VOCAB_INDEX.get(src.trim());
      if (directId !== undefined && directId > 0) {
        const w = DELIM_VOCAB_TOKENS.has(src.trim()) ? baseWeight * 0.35 : baseWeight;
        bias[directId] = Math.min(maxWeight, Math.max(bias[directId], w));
        hasAny = true;
        continue;
      }
      const subIds = tokenizeExprIntoIds(src);
      for (const tid of subIds) {
        if (tid <= 0) continue;
        bias[tid] = Math.min(maxWeight, Math.max(bias[tid], baseWeight * 0.72));
        hasAny = true;
      }
    }
  };

  applyTokens(expectedTokens, 0.72, 1.25);
  applyTokens(stemLatexSegments, 0.36, 0.95);
  bias[0] = 0;

  return hasAny ? bias : undefined;
}

function strokeBbox(pts: Point[]): [number, number, number, number, number, number] {
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const p of pts) {
    if (p.x < x0) x0 = p.x;
    if (p.x > x1) x1 = p.x;
    if (p.y < y0) y0 = p.y;
    if (p.y > y1) y1 = p.y;
  }
  const w = Math.max(1e-3, x1 - x0);
  const h = Math.max(1e-3, y1 - y0);
  return [x0, y0, w, h, x0 + 0.5 * w, y0 + 0.5 * h];
}

/**
 * 与 Python `mathwriting_loader.py#extract_pair_geom_10` 逐维一致的 10 维尺度不变二维笔画对特征。
 */
export function extractPairGeom10(strokeA: Stroke, strokeB: Stroke, exprScale: number): Float32Array {
  const out = new Float32Array(10);
  if (strokeA.points.length === 0 || strokeB.points.length === 0) {
    return out;
  }
  const s = Math.max(12, exprScale);
  const [ax0, ay0, aw, ah, acx, acy] = strokeBbox(strokeA.points);
  const [bx0, by0, bw, bh, bcx, bcy] = strokeBbox(strokeB.points);
  const interX = Math.max(0, Math.min(ax0 + aw, bx0 + bw) - Math.max(ax0, bx0));
  const interY = Math.max(0, Math.min(ay0 + ah, by0 + bh) - Math.max(ay0, by0));
  const xOv = interX / Math.max(1e-3, Math.min(aw, bw));
  const yOv = interY / Math.max(1e-3, Math.min(ah, bh));

  const gapX = Math.max(0, bx0 - (ax0 + aw), ax0 - (bx0 + bw));
  const gapY = Math.max(0, by0 - (ay0 + ah), ay0 - (by0 + bh));
  const minDist = Math.hypot(gapX, gapY) / s;
  const areaA = Math.max(1, aw * ah);
  const areaB = Math.max(1, bw * bh);
  const logArea = clamp(Math.log(areaB / areaA), -3, 3);

  out[0] = clamp((bcx - acx) / s, -3, 3);
  out[1] = clamp((bcy - acy) / s, -3, 3);
  out[2] = clamp(aw / s, 0, 3);
  out[3] = clamp(ah / s, 0, 3);
  out[4] = clamp(bw / s, 0, 3);
  out[5] = clamp(bh / s, 0, 3);
  out[6] = clamp(xOv, 0, 1);
  out[7] = clamp(yOv, 0, 1);
  out[8] = clamp(minDist, 0, 3);
  out[9] = logArea;
  return out;
}

/**
 * 与 Python `mathwriting_loader.py#arc_length_resample_strokes` 完全等价的零截断自适应等弧长重采样（支持 10 通道二维边框感知与 6 通道兼容）。
 */
export function arcLengthResampleStrokesTS(
  strokes: Stroke[],
  stepNorm = 0.085,
  maxLen = 192,
  includeEraser = false,
  spatialDim = 10,
): { data: Float32Array; seqLen: number } {
  const valid = strokes.filter(s => (includeEraser || s.tool !== 'eraser') && s.points.length >= 1);
  if (valid.length === 0) {
    return { data: new Float32Array(spatialDim), seqLen: 1 };
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const s of valid) {
    for (const p of s.points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
  }
  const scale = Math.max(12, maxY - minY, 0.25 * (maxX - minX));
  const normCenterX = (0.5 * (minX + maxX)) / scale;
  const normCenterY = (0.5 * (minY + maxY)) / scale;

  const normStrokes: Array<{
    pts: Point[];
    cum: number[];
    scx: number;
    scy: number;
    sw: number;
    sh: number;
  }> = [];
  let totalArcAll = 0;
  for (const s of valid) {
    const pts = s.points;
    let sx0 = Infinity;
    let sx1 = -Infinity;
    let sy0 = Infinity;
    let sy1 = -Infinity;
    for (const p of pts) {
      const nx = p.x / scale;
      const ny = p.y / scale;
      if (nx < sx0) sx0 = nx;
      if (nx > sx1) sx1 = nx;
      if (ny < sy0) sy0 = ny;
      if (ny > sy1) sy1 = ny;
    }
    const scx = clamp(0.5 * (sx0 + sx1) - normCenterX, -2.5, 2.5);
    const scy = clamp(0.5 * (sy0 + sy1) - normCenterY, -1.5, 1.5);
    const sw = clamp(sx1 - sx0, 0, 3.0);
    const sh = clamp(sy1 - sy0, 0, 3.0);

    const cum: number[] = [0];
    for (let i = 1; i < pts.length; i++) {
      const d = Math.hypot((pts[i].x - pts[i - 1].x) / scale, (pts[i].y - pts[i - 1].y) / scale);
      cum.push(cum[cum.length - 1] + d);
    }
    totalArcAll += cum[cum.length - 1];
    normStrokes.push({ pts, cum, scx, scy, sw, sh });
  }

  const budget = Math.max(8, maxLen - normStrokes.length - 2);
  const effStep = totalArcAll > 1e-4 ? Math.max(stepNorm, totalArcAll / budget) : stepNorm;

  const resampled: Array<{
    x: number;
    y: number;
    pen: number;
    scx: number;
    scy: number;
    sw: number;
    sh: number;
  }> = [];
  for (const { pts, cum, scx, scy, sw, sh } of normStrokes) {
    if (pts.length === 1) {
      resampled.push({ x: pts[0].x / scale, y: pts[0].y / scale, pen: 0, scx, scy, sw, sh });
      continue;
    }
    const totalLen = cum[cum.length - 1];
    if (totalLen < 1e-4) {
      resampled.push({ x: pts[0].x / scale, y: pts[0].y / scale, pen: 0, scx, scy, sw, sh });
      continue;
    }
    const nSteps = Math.max(2, Math.min(64, Math.ceil(totalLen / effStep) + 1));
    let segIdx = 0;
    for (let idxT = 0; idxT < nSteps; idxT++) {
      const targetS = (totalLen * idxT) / Math.max(1, nSteps - 1);
      while (segIdx + 1 < cum.length && cum[segIdx + 1] < targetS) {
        segIdx += 1;
      }
      const s0 = cum[segIdx];
      const s1 = cum[Math.min(segIdx + 1, cum.length - 1)];
      const p0 = pts[segIdx];
      const p1 = pts[Math.min(segIdx + 1, pts.length - 1)];
      const alpha = s1 > s0 ? (targetS - s0) / Math.max(1e-6, s1 - s0) : 0;
      const rx = (p0.x + alpha * (p1.x - p0.x)) / scale;
      const ry = (p0.y + alpha * (p1.y - p0.y)) / scale;
      resampled.push({
        x: rx,
        y: ry,
        pen: idxT === nSteps - 1 ? 0 : 1,
        scx,
        scy,
        sw,
        sh,
      });
    }
    if (resampled.length >= maxLen) {
      resampled.length = maxLen;
      break;
    }
  }

  const seqLen = Math.max(1, resampled.length);
  const out = new Float32Array(seqLen * spatialDim);
  for (let i = 0; i < resampled.length; i++) {
    const cur = resampled[i];
    const prev = i > 0 ? resampled[i - 1] : cur;
    const dx = i === 0 ? 0 : clamp(cur.x - prev.x, -4, 4);
    const dy = i === 0 ? 0 : clamp(cur.y - prev.y, -4, 4);
    const ds = Math.hypot(dx, dy);
    const cosT = ds > 1e-6 ? dx / ds : 1;
    const sinT = ds > 1e-6 ? dy / ds : 0;
    const yRel = clamp(cur.y - normCenterY, -1.5, 1.5);
    const off = i * spatialDim;
    out[off] = dx;
    out[off + 1] = dy;
    out[off + 2] = cosT;
    out[off + 3] = sinT;
    out[off + 4] = cur.pen;
    out[off + 5] = yRel;
    if (spatialDim >= 10) {
      out[off + 6] = cur.scx;
      out[off + 7] = cur.scy;
      out[off + 8] = cur.sw;
      out[off + 9] = cur.sh;
    }
  }
  return { data: out, seqLen };
}

/**
 * 纯 TypeScript Float32Array 端侧双流神经网络引擎（Expo Go 无需额外原生库即可直接运行真实权重）。
 *
 * 【自研推理核】数值严格对齐 PyTorch `EdgeMathGraphNet` (10-ch + StrokeGraphEdgeHead) 与 14 维 `CognitiveStateNet`。
 */
export class EdgeNeuralModelRunner {
  private readonly mathWeights = new Map<string, Float32Array>();
  private readonly cogWeights = new Map<string, Float32Array>();
  private readonly mathD: number;
  private readonly cogD: number;
  private readonly mathLayers: number;
  private readonly nHeads: number;
  private readonly maxSpatialLen: number;
  private readonly spatialDim: number;
  private readonly dynamicsDim: number;
  private readonly edgeGeomDim: number;
  private readonly edgeHiddenDim: number;
  private readonly edgeClasses: number;
  private loaded = false;

  constructor(bundle: EdgeNeuralBundle = EDGE_NEURAL_BUNDLE) {
    const cfg = bundle.config ?? {};
    this.cogD = Math.max(16, Number(cfg.d_model ?? 64));
    this.mathD = Math.max(16, Number(cfg.math_d_model ?? this.cogD));
    this.mathLayers = Math.max(1, Number(cfg.conformer_layers ?? 4));
    this.nHeads = Math.max(1, Number(cfg.n_heads ?? 4));
    this.maxSpatialLen = Math.max(32, Number(cfg.max_spatial_len ?? 192));
    this.edgeGeomDim = Math.max(1, Number(cfg.edge_geom_dim ?? 10));
    this.edgeHiddenDim = Math.max(8, Number(cfg.edge_hidden_dim ?? 48));
    this.edgeClasses = Math.max(2, Number(cfg.edge_classes ?? 7));

    const mathKeys = Object.keys(bundle.mathEncoder);
    const cogKeys = Object.keys(bundle.cognitiveNet);
    if (mathKeys.length === 0 || cogKeys.length === 0) {
      this.spatialDim = Number(cfg.spatial_in_dim ?? 10);
      this.dynamicsDim = Number(cfg.dynamics_in_dim ?? 14);
      return;
    }
    for (const k of mathKeys) {
      this.mathWeights.set(k, dequantizeTensor(bundle.mathEncoder[k]));
    }
    for (const k of cogKeys) {
      this.cogWeights.set(k, dequantizeTensor(bundle.cognitiveNet[k]));
    }
    const stemW = this.mathWeights.get('stem.0.weight');
    const dynW = this.cogWeights.get('dynamics_encoder.proj.0.weight');
    this.spatialDim = stemW ? Math.max(6, Math.round(stemW.length / this.mathD)) : Number(cfg.spatial_in_dim ?? 10);
    this.dynamicsDim = dynW ? Math.max(10, Math.round(dynW.length / this.cogD)) : Number(cfg.dynamics_in_dim ?? 14);
    this.loaded = this.mathWeights.has('stem.0.weight') && this.cogWeights.has('class_head.weight');
  }

  public isLoaded(): boolean {
    return this.loaded;
  }

  public getMathEmbedDim(): number {
    return this.mathD;
  }

  public getSpatialDim(): number {
    return this.spatialDim;
  }

  public getDynamicsDim(): number {
    return this.dynamicsDim;
  }

  public hasEdgeHead(): boolean {
    return (
      this.loaded &&
      this.mathWeights.has('edge_head.proj_u.weight') &&
      this.mathWeights.has('edge_head.mlp_1.weight')
    );
  }

  private mw(name: string): Float32Array {
    const t = this.mathWeights.get(name);
    if (!t) throw new Error(`Missing mathEncoder weight: ${name}`);
    return t;
  }

  private cw(name: string): Float32Array {
    const t = this.cogWeights.get(name);
    if (!t) throw new Error(`Missing cognitiveNet weight: ${name}`);
    return t;
  }

  public inferMathStrokes(
    strokes: Stroke[],
    maxLen = this.maxSpatialLen,
    includeEraser = false,
    decodeMode: 'beam' | 'greedy' | 'none' = 'beam',
    decodeOptions?: CtcDecodeOptions,
  ): NeuralMathOutput {
    if (!this.loaded) {
      return {
        recognizedLatex: '',
        semPooled: new Float32Array(this.mathD),
        tokenPosteriors: new Float32Array(MATH_VOCAB_SIZE),
      };
    }
    const { data, seqLen } = arcLengthResampleStrokesTS(
      strokes,
      0.085,
      maxLen,
      includeEraser,
      this.spatialDim,
    );
    const D = this.mathD;

    // 1. Stem: Linear(spatialDim->D) -> LayerNorm -> SiLU
    const stemLin = linear(
      data,
      seqLen,
      this.spatialDim,
      D,
      this.mw('stem.0.weight'),
      this.mw('stem.0.bias'),
    );
    layerNorm(stemLin, seqLen, D, this.mw('stem.1.weight'), this.mw('stem.1.bias'), true);
    siluInPlace(stemLin);

    // 2. ConvSubsample(stride=2, k=5) -> SiLU -> SubNorm
    const sub = conv1dStride2Pad2K5(
      stemLin,
      seqLen,
      D,
      D,
      this.mw('subsample.weight'),
      this.mw('subsample.bias'),
    );
    siluInPlace(sub.out);
    const h = layerNorm(sub.out, sub.outLen, D, this.mw('sub_norm.weight'), this.mw('sub_norm.bias'), true);
    const subLen = sub.outLen;

    // 3. 无参正弦位置编码（与 Python EdgeMathEncoder 逐维一致）
    const halfD = D / 2;
    for (let pos = 0; pos < subLen; pos++) {
      const pBase = pos * D;
      for (let k = 0; k < halfD; k++) {
        const freq = Math.exp(k * (-Math.log(10000) / halfD));
        const ang = pos * freq;
        h[pBase + k] += Math.sin(ang);
        h[pBase + halfD + k] += Math.cos(ang);
      }
    }

    // 4. 多层 Conformer Block（自动按导出配置迭代）
    for (let b = 0; b < this.mathLayers; b++) {
      const p = `blocks.${b}.`;
      const normAttn = layerNorm(h, subLen, D, this.mw(p + 'norm_attn.weight'), this.mw(p + 'norm_attn.bias'));
      const attnOut = multiheadAttention(
        normAttn,
        normAttn,
        normAttn,
        subLen,
        subLen,
        D,
        this.nHeads,
        this.mw(p + 'attn.in_proj_weight'),
        this.mw(p + 'attn.in_proj_bias'),
        this.mw(p + 'attn.out_proj.weight'),
        this.mw(p + 'attn.out_proj.bias'),
      );
      for (let i = 0; i < h.length; i++) h[i] += attnOut[i];

      const normConv = layerNorm(h, subLen, D, this.mw(p + 'norm_conv.weight'), this.mw(p + 'norm_conv.bias'));
      const pwIn = linear(normConv, subLen, D, D * 2, this.mw(p + 'pw_in.weight'), this.mw(p + 'pw_in.bias'));
      const glu = new Float32Array(subLen * D);
      for (let t = 0; t < subLen; t++) {
        const inBase = t * D * 2;
        const gBase = t * D;
        for (let d = 0; d < D; d++) {
          glu[gBase + d] = pwIn[inBase + d] * sigmoid(pwIn[inBase + D + d]);
        }
      }
      const dw = depthwiseConv1dK5Pad2(glu, subLen, D, this.mw(p + 'dw_conv.weight'), this.mw(p + 'dw_conv.bias'));
      siluInPlace(dw);
      const pwOut = linear(dw, subLen, D, D, this.mw(p + 'pw_out.weight'), this.mw(p + 'pw_out.bias'));
      for (let i = 0; i < h.length; i++) h[i] += pwOut[i];

      const normFfn = layerNorm(h, subLen, D, this.mw(p + 'norm_ffn.weight'), this.mw(p + 'norm_ffn.bias'));
      const ffn1 = linear(normFfn, subLen, D, D * 2, this.mw(p + 'ffn.0.weight'), this.mw(p + 'ffn.0.bias'));
      siluInPlace(ffn1);
      const ffn2 = linear(ffn1, subLen, D * 2, D, this.mw(p + 'ffn.3.weight'), this.mw(p + 'ffn.3.bias'));
      for (let i = 0; i < h.length; i++) h[i] += 0.5 * ffn2[i];
    }

    const zSemSeq = layerNorm(h, subLen, D, this.mw('out_norm.weight'), this.mw('out_norm.bias'), true);
    const semPooled = new Float32Array(D);
    for (let t = 0; t < subLen; t++) {
      const off = t * D;
      for (let d = 0; d < D; d++) semPooled[d] += zSemSeq[off + d];
    }
    for (let d = 0; d < D; d++) semPooled[d] /= Math.max(1, subLen);

    if (decodeMode === 'none') {
      return {
        recognizedLatex: '',
        semPooled,
        tokenPosteriors: new Float32Array(MATH_VOCAB_SIZE),
      };
    }

    const vocabSize = this.mw('ctc_head.bias').length;
    const ctcLogits = linear(zSemSeq, subLen, D, vocabSize, this.mw('ctc_head.weight'), this.mw('ctc_head.bias'));
    const tokenPosteriors = computeCtcTokenPosteriors(ctcLogits, subLen, vocabSize);

    if (decodeMode === 'beam') {
      return {
        recognizedLatex: decodeCtcBeamSearch(ctcLogits, subLen, vocabSize, 4, 6, decodeOptions),
        semPooled,
        tokenPosteriors,
      };
    }

    const predIds: number[] = [];
    for (let t = 0; t < subLen; t++) {
      const off = t * vocabSize;
      let bestId = 0;
      let bestVal = ctcLogits[off];
      for (let v = 1; v < vocabSize; v++) {
        if (ctcLogits[off + v] > bestVal) {
          bestVal = ctcLogits[off + v];
          bestId = v;
        }
      }
      predIds.push(bestId);
    }

    return {
      recognizedLatex: decodeCtcGreedy(predIds),
      semPooled,
      tokenPosteriors,
    };
  }

  /**
   * 对单条完整笔画提取语义嵌入（maxLen=24，跳过 CTC 解码以提速，与 PyTorch `_extract_step_semantic_embeddings` 对齐）。
   */
  public inferSingleStrokeSemantic(stroke: Stroke): Float32Array {
    if (!this.loaded || stroke.points.length === 0) {
      return new Float32Array(this.mathD);
    }
    return this.inferMathStrokes([stroke], 24, true, 'none').semPooled;
  }

  /**
   * StrokeGraphEdgeHead 双线性笔画关系分类头（与 PyTorch `StrokeGraphEdgeHead` 逐维一致）。
   * 预测 7 类二维关系：NONE / SAME_SYMBOL / RIGHT / ABOVE / BELOW / SUP / SUB。
   */
  public inferStrokeGraphEdges(
    semA: Float32Array,
    semB: Float32Array,
    geom10: Float32Array,
  ): StrokeEdgePrediction {
    if (!this.hasEdgeHead()) {
      return {
        bestClass: 0,
        relation: 'NONE',
        probs: [1, 0, 0, 0, 0, 0, 0],
      };
    }
    const semDim = this.mathD;
    const hDim = this.edgeHiddenDim;
    const gDim = this.edgeGeomDim;
    const nCls = this.edgeClasses;

    const u = linear(semA, 1, semDim, hDim, this.mw('edge_head.proj_u.weight'), this.mw('edge_head.proj_u.bias'));
    const v = linear(semB, 1, semDim, hDim, this.mw('edge_head.proj_v.weight'), this.mw('edge_head.proj_v.bias'));

    const feat = new Float32Array(hDim * 3 + gDim);
    feat.set(u, 0);
    feat.set(v, hDim);
    for (let i = 0; i < hDim; i++) {
      feat[hDim * 2 + i] = u[i] * v[i];
    }
    for (let i = 0; i < Math.min(gDim, geom10.length); i++) {
      feat[hDim * 3 + i] = geom10[i];
    }

    const hidden = linear(
      feat,
      1,
      hDim * 3 + gDim,
      hDim,
      this.mw('edge_head.mlp_0.weight'),
      this.mw('edge_head.mlp_0.bias'),
    );
    siluInPlace(hidden);

    const logits = linear(hidden, 1, hDim, nCls, this.mw('edge_head.mlp_1.weight'), this.mw('edge_head.mlp_1.bias'));
    let maxLogit = -Infinity;
    for (let c = 0; c < nCls; c++) {
      if (logits[c] > maxLogit) maxLogit = logits[c];
    }
    const probs = new Array<number>(nCls);
    let expSum = 0;
    for (let c = 0; c < nCls; c++) {
      probs[c] = Math.exp(logits[c] - maxLogit);
      expSum += probs[c];
    }
    let bestClass = 0;
    let bestProb = -1;
    for (let c = 0; c < nCls; c++) {
      probs[c] = Number((probs[c] / Math.max(1e-8, expSum)).toFixed(4));
      if (probs[c] > bestProb) {
        bestProb = probs[c];
        bestClass = c;
      }
    }
    const relation = (EDGE_RELATIONS[bestClass] as StrokeEdgeRelation) ?? 'NONE';
    return { bestClass, relation, probs };
  }

  public inferCognitiveWindow(
    dynamicsSteps: Float32Array,
    semStepSeq: Float32Array,
    winSteps: number,
  ): NeuralCognitiveOutput {
    const D = this.cogD;
    const targetDynDim = this.dynamicsDim;
    const inRowDim = winSteps > 0 ? Math.max(1, Math.floor(dynamicsSteps.length / winSteps)) : targetDynDim;
    let dynInput = dynamicsSteps;
    if (inRowDim !== targetDynDim) {
      dynInput = new Float32Array(winSteps * targetDynDim);
      const copyCols = Math.min(inRowDim, targetDynDim);
      for (let t = 0; t < winSteps; t++) {
        for (let c = 0; c < copyCols; c++) {
          dynInput[t * targetDynDim + c] = dynamicsSteps[t * inRowDim + c];
        }
      }
    }

    // 1. CausalDynamicsEncoder: proj(14->D) -> LayerNorm -> SiLU -> 2-layer GRU
    const proj = linear(
      dynInput,
      winSteps,
      targetDynDim,
      D,
      this.cw('dynamics_encoder.proj.0.weight'),
      this.cw('dynamics_encoder.proj.0.bias'),
    );
    layerNorm(
      proj,
      winSteps,
      D,
      this.cw('dynamics_encoder.proj.1.weight'),
      this.cw('dynamics_encoder.proj.1.bias'),
      true,
    );
    siluInPlace(proj);

    const gru0 = gruForward(
      proj,
      winSteps,
      D,
      this.cw('dynamics_encoder.gru.weight_ih_l0'),
      this.cw('dynamics_encoder.gru.weight_hh_l0'),
      this.cw('dynamics_encoder.gru.bias_ih_l0'),
      this.cw('dynamics_encoder.gru.bias_hh_l0'),
    );
    const zDyn = gruForward(
      gru0,
      winSteps,
      D,
      this.cw('dynamics_encoder.gru.weight_ih_l1'),
      this.cw('dynamics_encoder.gru.weight_hh_l1'),
      this.cw('dynamics_encoder.gru.bias_ih_l1'),
      this.cw('dynamics_encoder.gru.bias_hh_l1'),
    );

    // 2. 语义投影 (mathD -> cogD) + 因果末步 Cross-Attention（仅看历史 0..winSteps-1）
    const semK = this.cogWeights.has('sem_proj.weight')
      ? linear(semStepSeq, winSteps, this.mathD, D, this.cw('sem_proj.weight'), this.cw('sem_proj.bias'))
      : semStepSeq;
    const latestDyn = zDyn.subarray((winSteps - 1) * D, winSteps * D);
    const fused = multiheadAttention(
      latestDyn,
      semK,
      semK,
      1,
      winSteps,
      D,
      this.nHeads,
      this.cw('cross_attn.in_proj_weight'),
      this.cw('cross_attn.in_proj_bias'),
      this.cw('cross_attn.out_proj.weight'),
      this.cw('cross_attn.out_proj.bias'),
    );

    const combined = new Float32Array(D * 2);
    combined.set(latestDyn, 0);
    combined.set(fused, D);
    layerNorm(combined, 1, D * 2, this.cw('fuse_norm.weight'), this.cw('fuse_norm.bias'), true);

    const mlp = linear(combined, 1, D * 2, D, this.cw('mlp.0.weight'), this.cw('mlp.0.bias'));
    siluInPlace(mlp);

    const logits = linear(mlp, 1, D, 6, this.cw('class_head.weight'), this.cw('class_head.bias'));
    const surpRaw = linear(mlp, 1, D, 1, this.cw('surprise_head.weight'), this.cw('surprise_head.bias'));
    const mraRaw = linear(mlp, 1, D, 3, this.cw('mra_head.weight'), this.cw('mra_head.bias'));

    let maxLogit = -Infinity;
    for (let i = 0; i < 6; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    const probs = new Array<number>(6);
    let expSum = 0;
    for (let i = 0; i < 6; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      expSum += probs[i];
    }
    let bestCls = 0;
    let bestProb = -1;
    for (let i = 0; i < 6; i++) {
      probs[i] = Number((probs[i] / Math.max(1e-8, expSum)).toFixed(4));
      if (probs[i] > bestProb) {
        bestProb = probs[i];
        bestCls = i;
      }
    }

    return {
      classProbs: probs,
      predictedClass: bestCls,
      category: COGNITIVE_CATEGORIES[bestCls],
      surpriseScore: Number(sigmoid(surpRaw[0]).toFixed(3)),
      momentumM: Number(sigmoid(mraRaw[0]).toFixed(3)),
      resistanceR: Number(sigmoid(mraRaw[1]).toFixed(3)),
      alignmentA: Number(Math.tanh(mraRaw[2]).toFixed(3)),
    };
  }
}
