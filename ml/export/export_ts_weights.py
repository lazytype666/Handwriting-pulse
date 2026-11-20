"""
《笔迹脉搏》轨道 1 端侧轻量权重量化导出脚本（直供 Expo Go / React Native 纯 TS Float32Array 内核）。

作用：
1. 生成与 Python 词表索引严格 1-to-1 对齐的 `src/engine/mathVocab.ts`；
2. 将训练完的 PyTorch `EdgeMathEncoder` + `CognitiveStateNet` 执行对称 INT8 量化与 Base64 打包，
   输出紧凑的 `src/engine/edgeNeuralWeights.ts`（无需新增任何 npm 原生库即可真机推理真实权重）。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data_pipeline.vocab import MATH_VOCAB, VOCAB_SIZE  # noqa: E402
from ml.models.decoupled_dual_stream import MODEL_CONFIG_DEFAULTS, TORCH_AVAILABLE  # noqa: E402


def _quantize_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, tensor in sd.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        if arr.ndim <= 1 or arr.size <= 256:
            out[name] = {
                "dtype": "f32",
                "shape": list(arr.shape),
                "data": [round(float(v), 6) for v in arr.ravel()],
            }
        else:
            max_abs = float(np.max(np.abs(arr)))
            scale = max(max_abs / 127.0, 1e-8)
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
            b64 = base64.b64encode(q.tobytes()).decode("ascii")
            out[name] = {
                "dtype": "i8",
                "shape": list(arr.shape),
                "scale": round(scale, 8),
                "b64": b64,
            }
    return out


def emit_vocab_ts(out_path: Path) -> None:
    lines = [
        "/**",
        " * 全大学数学 CTC 符号词表（高数 / 线代 / 概率统计，自动由 ml/export/export_ts_weights.py 与 Python 对齐生成）。",
        " */",
        f"export const MATH_VOCAB: readonly string[] = {json.dumps(MATH_VOCAB, ensure_ascii=False)};",
        f"export const MATH_VOCAB_SIZE = {VOCAB_SIZE};",
        "",
        "function formatTokensToLatex(tokens: string[]): string {",
        "  const pieces: string[] = [];",
        "  for (const tok of tokens) {",
        "    const prevTok = pieces.length > 0 ? pieces[pieces.length - 1] : '';",
        "    if (tok.startsWith('\\\\') || (prevTok.startsWith('\\\\') && /^[a-z0-9]+$/i.test(tok))) {",
        "      pieces.push(pieces.length > 0 ? ' ' + tok : tok);",
        "    } else {",
        "      pieces.push(tok);",
        "    }",
        "  }",
        "  return pieces.join('').trim();",
        "}",
        "",
        "export function decodeCtcGreedy(predIds: number[]): string {",
        "  const tokens: string[] = [];",
        "  let prev = -1;",
        "  for (const idx of predIds) {",
        "    if (idx > 0 && idx !== prev && idx < MATH_VOCAB.length) {",
        "      tokens.push(MATH_VOCAB[idx]);",
        "    }",
        "    prev = idx;",
        "  }",
        "  return formatTokensToLatex(tokens);",
        "}",
        "",
        "function logAddExp(a: number, b: number): number {",
        "  if (a === -Infinity) return b;",
        "  if (b === -Infinity) return a;",
        "  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));",
        "}",
        "",
        "/**",
        " * 轻量 CTC Prefix Beam Search 解码（与 Python `decode_ctc_beam_search` 严格等价）。",
        " */",
        "export function decodeCtcBeamSearch(",
        "  ctcLogits: Float32Array,",
        "  frames: number,",
        "  vocabSize: number,",
        "  beamWidth = 4,",
        "  topK = 6,",
        "): string {",
        "  if (frames <= 0 || vocabSize <= 0) return '';",
        "  type BeamEntry = { ids: number[]; pb: number; pnb: number };",
        "  let beams = new Map<string, BeamEntry>([['', { ids: [], pb: 0, pnb: -Infinity }]]);",
        "  const logProbs = new Float32Array(vocabSize);",
        "  const kEff = Math.min(topK, vocabSize);",
        "  const topIds = new Int32Array(kEff);",
        "  const topVals = new Float32Array(kEff);",
        "",
        "  for (let t = 0; t < frames; t++) {",
        "    const off = t * vocabSize;",
        "    let maxV = -Infinity;",
        "    for (let v = 0; v < vocabSize; v++) {",
        "      const val = ctcLogits[off + v];",
        "      if (val > maxV) maxV = val;",
        "    }",
        "    let sumExp = 0;",
        "    for (let v = 0; v < vocabSize; v++) sumExp += Math.exp(ctcLogits[off + v] - maxV);",
        "    const logZ = maxV + Math.log(Math.max(1e-12, sumExp));",
        "    for (let v = 0; v < vocabSize; v++) logProbs[v] = ctcLogits[off + v] - logZ;",
        "",
        "    topIds.fill(0);",
        "    topVals.fill(-Infinity);",
        "    for (let v = 0; v < vocabSize; v++) {",
        "      const lp = logProbs[v];",
        "      if (lp > topVals[kEff - 1]) {",
        "        let pos = kEff - 1;",
        "        while (pos > 0 && lp > topVals[pos - 1]) {",
        "          topVals[pos] = topVals[pos - 1];",
        "          topIds[pos] = topIds[pos - 1];",
        "          pos--;",
        "        }",
        "        topVals[pos] = lp;",
        "        topIds[pos] = v;",
        "      }",
        "    }",
        "    const candidates: number[] = [];",
        "    let hasBlank = false;",
        "    for (let i = 0; i < kEff; i++) {",
        "      if (topIds[i] === 0) hasBlank = true;",
        "      candidates.push(topIds[i]);",
        "    }",
        "    if (!hasBlank) candidates.push(0);",
        "",
        "    const nextBeams = new Map<string, BeamEntry>();",
        "    const upsert = (key: string, ids: number[], addPb: number, addPnb: number) => {",
        "      const prev = nextBeams.get(key);",
        "      if (!prev) {",
        "        nextBeams.set(key, { ids, pb: addPb, pnb: addPnb });",
        "      } else {",
        "        if (addPb !== -Infinity) prev.pb = logAddExp(prev.pb, addPb);",
        "        if (addPnb !== -Infinity) prev.pnb = logAddExp(prev.pnb, addPnb);",
        "      }",
        "    };",
        "",
        "    for (const [key, entry] of beams.entries()) {",
        "      const pTot = logAddExp(entry.pb, entry.pnb);",
        "      const lastTok = entry.ids.length > 0 ? entry.ids[entry.ids.length - 1] : -1;",
        "      for (const idx of candidates) {",
        "        const lp = logProbs[idx];",
        "        if (idx === 0) {",
        "          upsert(key, entry.ids, pTot + lp, -Infinity);",
        "        } else if (idx === lastTok) {",
        "          upsert(key, entry.ids, -Infinity, entry.pnb + lp);",
        "          const extIds = [...entry.ids, idx];",
        "          const extKey = key ? `${key},${idx}` : String(idx);",
        "          upsert(extKey, extIds, -Infinity, entry.pb + lp);",
        "        } else {",
        "          const extIds = [...entry.ids, idx];",
        "          const extKey = key ? `${key},${idx}` : String(idx);",
        "          upsert(extKey, extIds, -Infinity, pTot + lp);",
        "        }",
        "      }",
        "    }",
        "",
        "    const ranked = Array.from(nextBeams.entries())",
        "      .sort((a, b) => logAddExp(b[1].pb, b[1].pnb) - logAddExp(a[1].pb, a[1].pnb))",
        "      .slice(0, beamWidth);",
        "    beams = new Map(ranked);",
        "  }",
        "",
        "  let bestIds: number[] = [];",
        "  let bestScore = -Infinity;",
        "  for (const entry of beams.values()) {",
        "    const sc = logAddExp(entry.pb, entry.pnb);",
        "    if (sc > bestScore) {",
        "      bestScore = sc;",
        "      bestIds = entry.ids;",
        "    }",
        "  }",
        "  const tokens: string[] = [];",
        "  for (const idx of bestIds) {",
        "    if (idx > 0 && idx < MATH_VOCAB.length) tokens.push(MATH_VOCAB[idx]);",
        "  }",
        "  return formatTokensToLatex(tokens);",
        "}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 INT8 量化权重至 src/engine/")
    parser.add_argument("--math-ckpt", type=str, default="ml/runs/stage1_edge_math_encoder.pt")
    parser.add_argument("--cog-ckpt", type=str, default="ml/runs/stage3_cognitive_state_net.pt")
    parser.add_argument("--vocab-out", type=str, default="src/engine/mathVocab.ts")
    parser.add_argument("--weights-out", type=str, default="src/engine/edgeNeuralWeights.ts")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit("当前环境无 PyTorch，请在 WSL2 环境中使用 /root/.inkpulse_venv/bin/python 运行。")

    import torch
    from ml.models.decoupled_dual_stream import CognitiveStateNet, EdgeMathEncoder

    math_d_model = int(MODEL_CONFIG_DEFAULTS.get("math_d_model", MODEL_CONFIG_DEFAULTS["d_model"]))
    math_enc = EdgeMathEncoder(
        in_dim=int(MODEL_CONFIG_DEFAULTS["spatial_in_dim"]),
        d_model=math_d_model,
        n_heads=int(MODEL_CONFIG_DEFAULTS["n_heads"]),
        num_layers=int(MODEL_CONFIG_DEFAULTS["conformer_layers"]),
        math_vocab_size=VOCAB_SIZE,
        dropout=0.0,
    )
    if Path(args.math_ckpt).exists():
        math_enc.load_state_dict(torch.load(args.math_ckpt, map_location="cpu"), strict=True)
    math_enc.eval()

    cog_net = CognitiveStateNet(MODEL_CONFIG_DEFAULTS)
    if Path(args.cog_ckpt).exists():
        cog_net.load_state_dict(torch.load(args.cog_ckpt, map_location="cpu"), strict=True)
    cog_net.eval()

    emit_vocab_ts(REPO_ROOT / args.vocab_out)

    bundle = {
        "config": MODEL_CONFIG_DEFAULTS,
        "mathEncoder": _quantize_state_dict(math_enc.state_dict()),
        "cognitiveNet": _quantize_state_dict(cog_net.state_dict()),
    }

    ts_content = "\n".join([
        "/**",
        " * 《笔迹脉搏》端侧双流神经网络量化权重快照（INT8 对称量化 Base64 打包，自动由 PyTorch 检查点导出）。",
        " * 严禁手工修改本文件。",
        " */",
        "",
        "export type QuantizedTensorSpec =",
        "  | { dtype: 'f32'; shape: number[]; data: number[] }",
        "  | { dtype: 'i8'; shape: number[]; scale: number; b64: string };",
        "",
        "export interface EdgeNeuralBundle {",
        "  config: Record<string, number>;",
        "  mathEncoder: Record<string, QuantizedTensorSpec>;",
        "  cognitiveNet: Record<string, QuantizedTensorSpec>;",
        "}",
        "",
        f"export const EDGE_NEURAL_BUNDLE: EdgeNeuralBundle = {json.dumps(bundle, separators=(',', ':'))};",
        "",
    ])
    weights_path = REPO_ROOT / args.weights_out
    weights_path.write_text(ts_content, encoding="utf-8")
    size_kb = weights_path.stat().st_size / 1024.0
    print(f"[TS 导出成功] 词表={args.vocab_out} | 量化权重包={args.weights_out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
