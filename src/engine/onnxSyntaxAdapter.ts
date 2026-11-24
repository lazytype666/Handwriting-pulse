import Constants from 'expo-constants';
import type { Stroke } from '../capture';
import {
  computeCtcTokenPosteriors,
  decodeCtcBeamSearch,
  formatTokensToLatex,
  MATH_VOCAB,
  MATH_VOCAB_SIZE,
  repairBalancedTokens,
  type CtcDecodeOptions,
} from './mathVocab';
import {
  arcLengthResampleStrokesTS,
  EdgeNeuralModelRunner,
  type NeuralMathOutput,
} from './neuralRuntime';
import type { IInkSyntaxEncoderAdapter } from './types';

export interface OrtTensorLike {
  data: Float32Array | Int32Array | BigInt64Array | Uint8Array;
  dims: readonly number[];
}

export interface OrtInferenceSessionLike {
  run(feeds: Record<string, OrtTensorLike>): Promise<Record<string, OrtTensorLike>>;
  release?(): Promise<void> | void;
}

export interface OrtRuntimeLike {
  InferenceSession: {
    create(pathOrBuffer: string | Uint8Array): Promise<OrtInferenceSessionLike>;
  };
  Tensor: new (
    type: 'float32' | 'int64' | 'bool',
    data: Float32Array | BigInt64Array | Uint8Array,
    dims: readonly number[],
  ) => OrtTensorLike;
}

export interface OnnxSyntaxAdapterConfig {
  /** EdgeMathGraphNet ONNX 模型本地文件路径（如 FileSystem.documentDirectory + 'edge_math_encoder.onnx'） */
  encoderModelPath?: string;
  /** 可选 CoMER Coverage Seq2Seq ONNX 解码器路径（edge_math_coverage_decoder.onnx） */
  coverageDecoderPath?: string;
  /** 可选直接注入 onnxruntime-react-native 模块实例（Dev Client / Release APK） */
  ortModule?: OrtRuntimeLike;
  /** 当未安装或处于 Expo Go 时，自动回退到纯 TS 量化推理核 */
  fallbackRunner?: EdgeNeuralModelRunner;
  /** Coverage Seq2Seq 最大自回归解码步数 */
  maxDecodeSteps?: number;
}

export interface OnnxDetailedEncodeResult extends NeuralMathOutput {
  alignmentScore: number;
  backend: 'onnx' | 'onnx_coverage' | 'ts_fallback';
}

const IS_EXPO_GO =
  Constants.appOwnership === 'expo' || Constants.executionEnvironment === 'storeClient';

const COVERAGE_SOS_ID = MATH_VOCAB_SIZE;
const COVERAGE_EOS_ID = MATH_VOCAB_SIZE + 1;
const COVERAGE_DEC_DIM = 192;
const COVERAGE_NUM_LAYERS = 2;

let registeredNativeOrt: OrtRuntimeLike | null = null;

/**
 * 在自定义 Dev Client / Release APK 入口中显式注册 `require('onnxruntime-react-native')`，
 * 避免在 Expo Go / Metro 无原生包环境下引发静态打包报错。
 */
export function registerOnnxNativeRuntime(ortModule: OrtRuntimeLike | null): void {
  registeredNativeOrt = ortModule;
}

function resolveAvailableOrt(explicit?: OrtRuntimeLike): OrtRuntimeLike | null {
  if (explicit) return explicit;
  if (registeredNativeOrt) return registeredNativeOrt;
  const g = globalThis as { __INKPULSE_ORT__?: OrtRuntimeLike };
  return g.__INKPULSE_ORT__ ?? null;
}

/**
 * Quote ④ 可选端侧 ONNX Runtime 原生加速适配器：
 * - Dev Client / APK（且注入 onnxruntime-react-native 并加载 `.onnx`）：执行 `edge_math_encoder.onnx`
 *   及可选 `edge_math_coverage_decoder.onnx`（CoMER Coverage Seq2Seq）；
 * - Expo Go 或无原生 ORT 环境：无感平滑回退到零原生依赖的 `EdgeNeuralModelRunner` (纯 TS INT8)。
 */
export class OnnxEdgeSyntaxAdapter implements IInkSyntaxEncoderAdapter {
  private ort: OrtRuntimeLike | null = null;
  private encoderSession: OrtInferenceSessionLike | null = null;
  private coverageSession: OrtInferenceSessionLike | null = null;
  private fallbackRunner: EdgeNeuralModelRunner;
  private maxDecodeSteps: number;
  private initPromise: Promise<boolean> | null = null;

  constructor(private readonly config: OnnxSyntaxAdapterConfig = {}) {
    this.fallbackRunner = config.fallbackRunner ?? new EdgeNeuralModelRunner();
    this.maxDecodeSteps = Math.max(8, config.maxDecodeSteps ?? 48);
  }

  public isNativeRuntimeAvailable(): boolean {
    return !IS_EXPO_GO && resolveAvailableOrt(this.config.ortModule) !== null;
  }

  public isLoaded(): boolean {
    return this.encoderSession !== null || this.fallbackRunner.isLoaded();
  }

  public hasOnnxSession(): boolean {
    return this.encoderSession !== null;
  }

  public hasCoverageDecoder(): boolean {
    return this.encoderSession !== null && this.coverageSession !== null;
  }

  public async initialize(): Promise<boolean> {
    if (this.encoderSession) return true;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      if (IS_EXPO_GO) {
        return false;
      }
      const ort = resolveAvailableOrt(this.config.ortModule);
      if (!ort || !this.config.encoderModelPath) {
        return false;
      }
      try {
        this.ort = ort;
        this.encoderSession = await ort.InferenceSession.create(this.config.encoderModelPath);
        if (this.config.coverageDecoderPath) {
          try {
            this.coverageSession = await ort.InferenceSession.create(this.config.coverageDecoderPath);
          } catch {
            this.coverageSession = null;
          }
        }
        return true;
      } catch {
        this.encoderSession = null;
        this.coverageSession = null;
        return false;
      }
    })();

    return this.initPromise;
  }

  public async dispose(): Promise<void> {
    try {
      await this.encoderSession?.release?.();
      await this.coverageSession?.release?.();
    } finally {
      this.encoderSession = null;
      this.coverageSession = null;
      this.initPromise = null;
    }
  }

  public async encodeRecentStrokes(strokes: Stroke[]): Promise<{
    recognizedLatex: string;
    alignmentScore: number;
  }> {
    const detailed = await this.encodeRecentStrokesDetailed(strokes);
    return {
      recognizedLatex: detailed.recognizedLatex,
      alignmentScore: detailed.alignmentScore,
    };
  }

  public async encodeRecentStrokesDetailed(
    strokes: Stroke[],
    decodeOptions?: CtcDecodeOptions,
  ): Promise<OnnxDetailedEncodeResult> {
    if (!this.encoderSession && this.isNativeRuntimeAvailable() && this.config.encoderModelPath) {
      await this.initialize();
    }

    const ort = this.ort;
    const enc = this.encoderSession;
    if (!ort || !enc) {
      const tsOut = this.fallbackRunner.inferMathStrokes(strokes, 192, false, 'beam', decodeOptions);
      return {
        ...tsOut,
        alignmentScore: 0,
        backend: 'ts_fallback',
      };
    }

    const { data, seqLen } = arcLengthResampleStrokesTS(strokes, 0.085, 192, false, 10);
    const padMask = new Uint8Array(seqLen);
    const encFeeds: Record<string, OrtTensorLike> = {
      spatial_seq: new ort.Tensor('float32', data, [1, seqLen, 10]),
      pad_mask: new ort.Tensor('bool', padMask, [1, seqLen]),
    };

    const encOutputs = await enc.run(encFeeds);
    const ctcTensor = encOutputs.ctc_logits;
    const zSeqTensor = encOutputs.z_sem_seq;
    const pooledTensor = encOutputs.sem_pooled;

    const ctcData = new Float32Array(ctcTensor.data as Float32Array);
    const subLen = ctcTensor.dims[1] ?? Math.max(1, Math.floor((seqLen + 4 - 5) / 2) + 1);
    const vocabSize = ctcTensor.dims[2] ?? MATH_VOCAB_SIZE;
    const semPooled = new Float32Array(pooledTensor.data as Float32Array);
    const tokenPosteriors = computeCtcTokenPosteriors(ctcData, subLen, vocabSize);

    // 若同时挂载了 CoMER Coverage Attention ONNX 解码器，运行自回归公式解码（优先抗长式漏字/重复）
    if (this.coverageSession && zSeqTensor) {
      try {
        const covLatex = await this.runCoverageDecoder(
          ort,
          this.coverageSession,
          new Float32Array(zSeqTensor.data as Float32Array),
          subLen,
          zSeqTensor.dims[2] ?? 96,
          decodeOptions,
        );
        if (covLatex) {
          return {
            recognizedLatex: covLatex,
            semPooled,
            tokenPosteriors,
            alignmentScore: 0,
            backend: 'onnx_coverage',
          };
        }
      } catch {
        // 回退到 CTC Beam Search
      }
    }

    const recognizedLatex = decodeCtcBeamSearch(ctcData, subLen, vocabSize, 4, 6, decodeOptions);
    return {
      recognizedLatex,
      semPooled,
      tokenPosteriors,
      alignmentScore: 0,
      backend: 'onnx',
    };
  }

  private async runCoverageDecoder(
    ort: OrtRuntimeLike,
    covSession: OrtInferenceSessionLike,
    zSemSeq: Float32Array,
    subLen: number,
    encDim: number,
    decodeOptions?: CtcDecodeOptions,
  ): Promise<string> {
    let prevToken = BigInt(COVERAGE_SOS_ID);
    let hidden = new Float32Array(COVERAGE_NUM_LAYERS * COVERAGE_DEC_DIM);
    let covAccum = new Float32Array(subLen);
    const zTensor = new ort.Tensor('float32', zSemSeq, [1, subLen, encDim]);

    const tokenIds: number[] = [];
    const priorBias = decodeOptions?.priorLogitBias;
    const enforceGrammar = decodeOptions?.enforceGrammar ?? true;

    for (let step = 0; step < this.maxDecodeSteps; step++) {
      const out = await covSession.run({
        prev_token_id: new ort.Tensor('int64', BigInt64Array.from([prevToken]), [1]),
        prev_hidden: new ort.Tensor('float32', hidden, [COVERAGE_NUM_LAYERS, 1, COVERAGE_DEC_DIM]),
        z_sem_seq: zTensor,
        cov_accum: new ort.Tensor('float32', covAccum, [1, subLen]),
      });

      const stepLogits = out.step_logits.data as Float32Array;
      hidden = new Float32Array(out.next_hidden.data as Float32Array);
      covAccum = new Float32Array(out.next_cov.data as Float32Array);

      let bestId = COVERAGE_EOS_ID;
      let bestScore = -Infinity;
      const numCls = stepLogits.length;
      for (let v = 1; v < numCls; v++) {
        if (v === COVERAGE_SOS_ID) continue;
        let score = stepLogits[v];
        if (priorBias && v < priorBias.length) {
          score += priorBias[v];
        }
        if (v === COVERAGE_EOS_ID && step < 1) {
          score -= 3.5;
        }
        if (score > bestScore) {
          bestScore = score;
          bestId = v;
        }
      }

      if (bestId === COVERAGE_EOS_ID || bestId === 0) {
        break;
      }
      if (bestId > 0 && bestId < MATH_VOCAB_SIZE) {
        tokenIds.push(bestId);
      }
      prevToken = BigInt(bestId);
    }

    const finalIds = enforceGrammar ? repairBalancedTokens(tokenIds) : tokenIds;
    const finalTokens = finalIds
      .filter(id => id > 0 && id < MATH_VOCAB.length)
      .map(id => MATH_VOCAB[id]);
    return formatTokensToLatex(finalTokens);
  }
}
