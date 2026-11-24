"""
《笔迹脉搏》轨道 1 端侧轻量权重量化导出脚本（直供 Expo Go / React Native 纯 TS Float32Array 内核）。

作用：
1. 生成与 Python 词表/边关系索引严格 1-to-1 对齐的 `src/engine/mathVocab.ts`（含 TAMER 语法栈约束 + 边缘后验计算）；
2. 将训练完的 PyTorch `EdgeMathGraphNet`（含 `StrokeGraphEdgeHead`）+ `CognitiveStateNet` 执行对称 INT8 量化与 Base64 打包，
   输出紧凑的 `src/engine/edgeNeuralWeights.ts`。
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

from ml.data_pipeline.vocab import EDGE_RELATIONS, MATH_VOCAB, VOCAB_SIZE  # noqa: E402
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
        " * 全大学数学 CTC 符号词表与二维笔画关系词表（自动由 ml/export/export_ts_weights.py 与 Python 对齐生成）。",
        " */",
        f"export const MATH_VOCAB: readonly string[] = {json.dumps(MATH_VOCAB, ensure_ascii=False)};",
        f"export const MATH_VOCAB_SIZE = {VOCAB_SIZE};",
        "",
        "export const MATH_VOCAB_INDEX: ReadonlyMap<string, number> = new Map(",
        "  MATH_VOCAB.map((tok, idx) => [tok, idx]),",
        ");",
        "",
        f"export const EDGE_RELATIONS: readonly string[] = {json.dumps(EDGE_RELATIONS, ensure_ascii=False)};",
        "export type StrokeEdgeRelation = 'NONE' | 'SAME_SYMBOL' | 'RIGHT' | 'ABOVE' | 'BELOW' | 'SUP' | 'SUB';",
        "",
        "export interface CtcDecodeOptions {",
        "  /** 题谱先验浅融合偏置向量（长度 125，按当前题干与步骤期望符号赋予对数概率加成） */",
        "  priorLogitBias?: Float32Array;",
        "  /** 是否启用 TAMER 风格的括号/上下标语法栈约束与树闭合修复 */",
        "  enforceGrammar?: boolean;",
        "}",
        "",
        "const ID_LPAREN = MATH_VOCAB.indexOf('(');",
        "const ID_RPAREN = MATH_VOCAB.indexOf(')');",
        "const ID_LBRACK = MATH_VOCAB.indexOf('[');",
        "const ID_RBRACK = MATH_VOCAB.indexOf(']');",
        "const ID_LBRACE = MATH_VOCAB.indexOf('{');",
        "const ID_RBRACE = MATH_VOCAB.indexOf('}');",
        "const ID_SUB = MATH_VOCAB.indexOf('_');",
        "const ID_SUP = MATH_VOCAB.indexOf('^');",
        "",
        "/**",
        " * 从 CTC 帧级 Logits (frames × vocabSize) 提取 125 维符号边缘后验概率 P_ctc(v) ∈ [0, 1]^125。",
        " * 供下游解题图谱步骤 Logits 与易错点检测 Logits 执行连续软概率内积。",
        " */",
        "export function computeCtcTokenPosteriors(",
        "  ctcLogits: Float32Array,",
        "  frames: number,",
        "  vocabSize: number,",
        "): Float32Array {",
        "  const posteriors = new Float32Array(vocabSize);",
        "  if (frames <= 0 || vocabSize <= 0) return posteriors;",
        "  const logNotSeen = new Float32Array(vocabSize);",
        "  const maxFrameProb = new Float32Array(vocabSize);",
        "  for (let t = 0; t < frames; t++) {",
        "    const off = t * vocabSize;",
        "    let maxV = -Infinity;",
        "    for (let v = 0; v < vocabSize; v++) {",
        "      const val = ctcLogits[off + v];",
        "      if (val > maxV) maxV = val;",
        "    }",
        "    let sumExp = 0;",
        "    for (let v = 0; v < vocabSize; v++) sumExp += Math.exp(ctcLogits[off + v] - maxV);",
        "    const invZ = 1 / Math.max(1e-12, sumExp);",
        "    for (let v = 1; v < vocabSize; v++) {",
        "      const p = Math.min(0.9999, Math.max(0, Math.exp(ctcLogits[off + v] - maxV) * invZ));",
        "      if (p > maxFrameProb[v]) maxFrameProb[v] = p;",
        "      logNotSeen[v] += Math.log(Math.max(1e-6, 1 - p * 0.65));",
        "    }",
        "  }",
        "  for (let v = 1; v < vocabSize; v++) {",
        "    const noisyOr = 1 - Math.exp(logNotSeen[v]);",
        "    posteriors[v] = Number(Math.min(1, 0.55 * maxFrameProb[v] + 0.45 * noisyOr).toFixed(4));",
        "  }",
        "  return posteriors;",
        "}",
        "",
        "function grammarTransitionPenalty(prevIds: number[], nextId: number): number {",
        "  const prevId = prevIds.length > 0 ? prevIds[prevIds.length - 1] : -1;",
        "  if ((nextId === ID_SUP || nextId === ID_SUB) && (prevId === ID_SUP || prevId === ID_SUB)) {",
        "    return -3.2;",
        "  }",
        "  if (",
        "    prevIds.length === 0 &&",
        "    (nextId === ID_SUP || nextId === ID_SUB || nextId === ID_RPAREN || nextId === ID_RBRACK || nextId === ID_RBRACE)",
        "  ) {",
        "    return -3.5;",
        "  }",
        "  if (nextId === ID_RPAREN || nextId === ID_RBRACK || nextId === ID_RBRACE) {",
        "    const openTarget = nextId === ID_RPAREN ? ID_LPAREN : nextId === ID_RBRACK ? ID_LBRACK : ID_LBRACE;",
        "    let depth = 0;",
        "    for (const id of prevIds) {",
        "      if (id === openTarget) depth += 1;",
        "      else if (id === nextId) depth -= 1;",
        "    }",
        "    if (depth <= 0) return -3.4;",
        "    if (prevId === openTarget) return -2.4;",
        "  }",
        "  return 0;",
        "}",
        "",
        "function computeUnclosedDelimiterCount(ids: number[]) {",
        "  let braceDepth = 0;",
        "  let parenDepth = 0;",
        "  let bracketDepth = 0;",
        "  for (const id of ids) {",
        "    if (id === ID_LBRACE) braceDepth = Math.max(0, braceDepth + 1);",
        "    else if (id === ID_RBRACE) braceDepth = Math.max(0, braceDepth - 1);",
        "    else if (id === ID_LPAREN) parenDepth = Math.max(0, parenDepth + 1);",
        "    else if (id === ID_RPAREN) parenDepth = Math.max(0, parenDepth - 1);",
        "    else if (id === ID_LBRACK) bracketDepth = Math.max(0, bracketDepth + 1);",
        "    else if (id === ID_RBRACK) bracketDepth = Math.max(0, bracketDepth - 1);",
        "  }",
        "  const last = ids.length > 0 ? ids[ids.length - 1] : -1;",
        "  return { braceDepth, parenDepth, bracketDepth, danglingScript: last === ID_SUP || last === ID_SUB };",
        "}",
        "",
        "export function repairBalancedTokens(ids: number[]): number[] {",
        "  const out = [...ids];",
        "  while (out.length > 0 && (out[out.length - 1] === ID_SUP || out[out.length - 1] === ID_SUB)) out.pop();",
        "  const st = computeUnclosedDelimiterCount(out);",
        "  for (let i = 0; i < st.braceDepth && i < 3; i++) out.push(ID_RBRACE);",
        "  for (let i = 0; i < st.parenDepth && i < 2; i++) out.push(ID_RPAREN);",
        "  for (let i = 0; i < st.bracketDepth && i < 2; i++) out.push(ID_RBRACK);",
        "  return out;",
        "}",
        "",
        "export function formatTokensToLatex(tokens: string[]): string {",
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
        "    if (idx > 0 && idx !== prev && idx < MATH_VOCAB.length) tokens.push(MATH_VOCAB[idx]);",
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
        " * 轻量 CTC Prefix Beam Search 解码（与 Python `decode_ctc_beam_search` 严格等价，含 TAMER 语法栈与题谱先验浅融合）。",
        " */",
        "export function decodeCtcBeamSearch(",
        "  ctcLogits: Float32Array,",
        "  frames: number,",
        "  vocabSize: number,",
        "  beamWidth = 4,",
        "  topK = 6,",
        "  options?: CtcDecodeOptions,",
        "): string {",
        "  if (frames <= 0 || vocabSize <= 0) return '';",
        "  type BeamEntry = { ids: number[]; pb: number; pnb: number };",
        "  let beams = new Map<string, BeamEntry>([['', { ids: [], pb: 0, pnb: -Infinity }]]);",
        "  const logProbs = new Float32Array(vocabSize);",
        "  const kEff = Math.min(topK, vocabSize);",
        "  const topIds = new Int32Array(kEff);",
        "  const topVals = new Float32Array(kEff);",
        "  const prior = options?.priorLogitBias && options.priorLogitBias.length === vocabSize ? options.priorLogitBias : null;",
        "  const useGrammar = Boolean(options?.enforceGrammar);",
        "",
        "  for (let t = 0; t < frames; t++) {",
        "    const off = t * vocabSize;",
        "    let maxV = -Infinity;",
        "    for (let v = 0; v < vocabSize; v++) {",
        "      const val = ctcLogits[off + v] + (prior && v > 0 ? prior[v] : 0);",
        "      if (val > maxV) maxV = val;",
        "    }",
        "    let sumExp = 0;",
        "    for (let v = 0; v < vocabSize; v++) {",
        "      sumExp += Math.exp(ctcLogits[off + v] + (prior && v > 0 ? prior[v] : 0) - maxV);",
        "    }",
        "    const logZ = maxV + Math.log(Math.max(1e-12, sumExp));",
        "    for (let v = 0; v < vocabSize; v++) {",
        "      logProbs[v] = ctcLogits[off + v] + (prior && v > 0 ? prior[v] : 0) - logZ;",
        "    }",
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
        "      if (!prev) nextBeams.set(key, { ids, pb: addPb, pnb: addPnb });",
        "      else {",
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
        "          const gPen = useGrammar ? grammarTransitionPenalty(entry.ids, idx) : 0;",
        "          upsert(key ? `${key},${idx}` : String(idx), [...entry.ids, idx], -Infinity, entry.pb + lp + gPen);",
        "        } else {",
        "          const gPen = useGrammar ? grammarTransitionPenalty(entry.ids, idx) : 0;",
        "          upsert(key ? `${key},${idx}` : String(idx), [...entry.ids, idx], -Infinity, pTot + lp + gPen);",
        "        }",
        "      }",
        "    }",
        "    const ranked = Array.from(nextBeams.entries())",
        "      .sort((a, b) => logAddExp(b[1].pb, b[1].pnb) - logAddExp(a[1].pb, a[1].pnb))",
        "      .slice(0, beamWidth);",
        "    beams = new Map(ranked);",
        "  }",
        "",
        "  let bestIds: number[] = [];",
        "  let bestScore = -Infinity;",
        "  for (const entry of beams.values()) {",
        "    let sc = logAddExp(entry.pb, entry.pnb);",
        "    if (useGrammar) {",
        "      const unclosed = computeUnclosedDelimiterCount(entry.ids);",
        "      sc -= (unclosed.braceDepth + unclosed.parenDepth + unclosed.bracketDepth) * 1.15 + (unclosed.danglingScript ? 1.8 : 0);",
        "    }",
        "    if (sc > bestScore) {",
        "      bestScore = sc;",
        "      bestIds = entry.ids;",
        "    }",
        "  }",
        "  const finalIds = useGrammar ? repairBalancedTokens(bestIds) : bestIds;",
        "  const tokens: string[] = [];",
        "  for (const idx of finalIds) {",
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
    from ml.models.decoupled_dual_stream import CognitiveStateNet, EdgeMathGraphNet

    math_d_model = int(MODEL_CONFIG_DEFAULTS["math_d_model"])
    math_enc = EdgeMathGraphNet(
        in_dim=int(MODEL_CONFIG_DEFAULTS["spatial_in_dim"]),
        d_model=math_d_model,
        n_heads=int(MODEL_CONFIG_DEFAULTS["n_heads"]),
        num_layers=int(MODEL_CONFIG_DEFAULTS["conformer_layers"]),
        math_vocab_size=VOCAB_SIZE,
        edge_geom_dim=int(MODEL_CONFIG_DEFAULTS["edge_geom_dim"]),
        edge_hidden_dim=int(MODEL_CONFIG_DEFAULTS["edge_hidden_dim"]),
        edge_classes=int(MODEL_CONFIG_DEFAULTS["edge_classes"]),
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
