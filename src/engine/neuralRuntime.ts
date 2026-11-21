import type { Point, Stroke } from '../capture';
import {
  EDGE_NEURAL_BUNDLE,
  type EdgeNeuralBundle,
  type QuantizedTensorSpec,
} from './edgeNeuralWeights';
import { decodeCtcBeamSearch, decodeCtcGreedy } from './mathVocab';

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

/**
 * 与 Python `mathwriting_loader.py#arc_length_resample_strokes` 完全等价的零截断自适应等弧长 6 通道重采样。
 */
export function arcLengthResampleStrokesTS(
  strokes: Stroke[],
  stepNorm = 0.085,
  maxLen = 192,
  includeEraser = false,
): { data: Float32Array; seqLen: number } {
  const valid = strokes.filter(s => (includeEraser || s.tool !== 'eraser') && s.points.length >= 1);
  if (valid.length === 0) {
    return { data: new Float32Array(6), seqLen: 1 };
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
  const normCenterY = (0.5 * (minY + maxY)) / scale;

  const normStrokes: Array<{ pts: Point[]; cum: number[] }> = [];
  let totalArcAll = 0;
  for (const s of valid) {
    const pts = s.points;
    const cum: number[] = [0];
    for (let i = 1; i < pts.length; i++) {
      const d = Math.hypot((pts[i].x - pts[i - 1].x) / scale, (pts[i].y - pts[i - 1].y) / scale);
      cum.push(cum[cum.length - 1] + d);
    }
    totalArcAll += cum[cum.length - 1];
    normStrokes.push({ pts, cum });
  }

  const budget = Math.max(8, maxLen - normStrokes.length - 2);
  const effStep = totalArcAll > 1e-4 ? Math.max(stepNorm, totalArcAll / budget) : stepNorm;

  const resampled: Array<{ x: number; y: number; pen: number }> = [];
  for (const { pts, cum } of normStrokes) {
    if (pts.length === 1) {
      resampled.push({ x: pts[0].x / scale, y: pts[0].y / scale, pen: 0 });
      continue;
    }
    const totalLen = cum[cum.length - 1];
    if (totalLen < 1e-4) {
      resampled.push({ x: pts[0].x / scale, y: pts[0].y / scale, pen: 0 });
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
      resampled.push({ x: rx, y: ry, pen: idxT === nSteps - 1 ? 0 : 1 });
    }
    if (resampled.length >= maxLen) {
      resampled.length = maxLen;
      break;
    }
  }

  const seqLen = Math.max(1, resampled.length);
  const out = new Float32Array(seqLen * 6);
  for (let i = 0; i < resampled.length; i++) {
    const cur = resampled[i];
    const prev = i > 0 ? resampled[i - 1] : cur;
    const dx = i === 0 ? 0 : clamp(cur.x - prev.x, -4, 4);
    const dy = i === 0 ? 0 : clamp(cur.y - prev.y, -4, 4);
    const ds = Math.hypot(dx, dy);
    const cosT = ds > 1e-6 ? dx / ds : 1;
    const sinT = ds > 1e-6 ? dy / ds : 0;
    const yRel = clamp(cur.y - normCenterY, -1.5, 1.5);
    const off = i * 6;
    out[off] = dx;
    out[off + 1] = dy;
    out[off + 2] = cosT;
    out[off + 3] = sinT;
    out[off + 4] = cur.pen;
    out[off + 5] = yRel;
  }
  return { data: out, seqLen };
}

/**
 * 纯 TypeScript Float32Array 端侧双流神经网络引擎（Expo Go 无需额外原生库即可直接运行真实权重）。
 *
 * 【自研推理核】数值严格对齐 PyTorch `EdgeMathEncoder` 与 `CognitiveStateNet`。
 */
export class EdgeNeuralModelRunner {
  private readonly mathWeights = new Map<string, Float32Array>();
  private readonly cogWeights = new Map<string, Float32Array>();
  private readonly mathD: number;
  private readonly cogD: number;
  private readonly mathLayers: number;
  private readonly nHeads: number;
  private readonly maxSpatialLen: number;
  private loaded = false;

  constructor(bundle: EdgeNeuralBundle = EDGE_NEURAL_BUNDLE) {
    const cfg = bundle.config ?? {};
    this.cogD = Math.max(16, Number(cfg.d_model ?? 64));
    this.mathD = Math.max(16, Number(cfg.math_d_model ?? this.cogD));
    this.mathLayers = Math.max(1, Number(cfg.conformer_layers ?? 4));
    this.nHeads = Math.max(1, Number(cfg.n_heads ?? 4));
    this.maxSpatialLen = Math.max(32, Number(cfg.max_spatial_len ?? 192));

    const mathKeys = Object.keys(bundle.mathEncoder);
    const cogKeys = Object.keys(bundle.cognitiveNet);
    if (mathKeys.length === 0 || cogKeys.length === 0) {
      return;
    }
    for (const k of mathKeys) {
      this.mathWeights.set(k, dequantizeTensor(bundle.mathEncoder[k]));
    }
    for (const k of cogKeys) {
      this.cogWeights.set(k, dequantizeTensor(bundle.cognitiveNet[k]));
    }
    this.loaded = this.mathWeights.has('stem.0.weight') && this.cogWeights.has('class_head.weight');
  }

  public isLoaded(): boolean {
    return this.loaded;
  }

  public getMathEmbedDim(): number {
    return this.mathD;
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
  ): NeuralMathOutput {
    if (!this.loaded) {
      return { recognizedLatex: '', semPooled: new Float32Array(this.mathD) };
    }
    const { data, seqLen } = arcLengthResampleStrokesTS(strokes, 0.085, maxLen, includeEraser);
    const D = this.mathD;

    // 1. Stem: Linear(6->D) -> LayerNorm -> SiLU
    const stemLin = linear(data, seqLen, 6, D, this.mw('stem.0.weight'), this.mw('stem.0.bias'));
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
    let h = layerNorm(sub.out, sub.outLen, D, this.mw('sub_norm.weight'), this.mw('sub_norm.bias'), true);
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
      return { recognizedLatex: '', semPooled };
    }

    const vocabSize = this.mw('ctc_head.bias').length;
    const ctcLogits = linear(zSemSeq, subLen, D, vocabSize, this.mw('ctc_head.weight'), this.mw('ctc_head.bias'));

    if (decodeMode === 'beam') {
      return {
        recognizedLatex: decodeCtcBeamSearch(ctcLogits, subLen, vocabSize, 4, 6),
        semPooled,
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

    return { recognizedLatex: decodeCtcGreedy(predIds), semPooled };
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

  public inferCognitiveWindow(
    dynamicsSteps: Float32Array,
    semStepSeq: Float32Array,
    winSteps: number,
  ): NeuralCognitiveOutput {
    const D = this.cogD;
    // 1. CausalDynamicsEncoder: proj(10->D) -> LayerNorm -> SiLU -> 2-layer GRU
    const proj = linear(
      dynamicsSteps,
      winSteps,
      10,
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
