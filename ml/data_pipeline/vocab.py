"""
《笔迹脉搏》全大学数学 CTC 符号词表与二维笔画图关系词表（高数 + 线性代数 + 概率论与数理统计）。

词表索引 0 固定保留为 `<blank>`（CTC 解码空白符），共 125 个 Token。
与移动端 `src/engine/mathVocab.ts` 保持严格 1-to-1 索引一致，保证 PyTorch/ONNX 与端侧 Float32Array 解码零偏差。
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CTC_BLANK = "<blank>"

DIGITS = list("0123456789")
LOWER_LETTERS = list("abcdefghijklmnopqrstuvwxyz")
UPPER_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
OPERATORS_AND_DELIMS = [
    "+", "-", "=", "(", ")", "[", "]", "{", "}", "|", "/", ",", ".", "_", "^", "~", "<", ">", "!"
]

LATEX_COMMANDS = [
    "\\frac",
    "\\sqrt",
    "\\int",
    "\\oint",
    "\\sum",
    "\\prod",
    "\\partial",
    "\\nabla",
    "\\infty",
    "\\lim",
    "\\to",
    "\\Rightarrow",
    "\\equiv",
    "\\neq",
    "\\approx",
    "\\le",
    "\\ge",
    "\\times",
    "\\cdot",
    "\\pm",
    "\\alpha",
    "\\beta",
    "\\gamma",
    "\\delta",
    "\\varepsilon",
    "\\theta",
    "\\lambda",
    "\\mu",
    "\\rho",
    "\\sigma",
    "\\phi",
    "\\Phi",
    "\\pi",
    "\\Omega",
    "\\Delta",
    "\\Lambda",
    "\\cap",
    "\\cup",
    "\\varnothing",
    "\\bar",
    "\\hat",
    "\\det",
    "\\matrix",
]

MATH_VOCAB: List[str] = (
    [CTC_BLANK] + DIGITS + LOWER_LETTERS + UPPER_LETTERS + OPERATORS_AND_DELIMS + LATEX_COMMANDS
)

TOKEN_TO_ID: Dict[str, int] = {tok: idx for idx, tok in enumerate(MATH_VOCAB)}
VOCAB_SIZE = len(MATH_VOCAB)

EDGE_RELATIONS: List[str] = ["NONE", "SAME_SYMBOL", "RIGHT", "ABOVE", "BELOW", "SUP", "SUB"]
EDGE_REL_TO_ID: Dict[str, int] = {rel: idx for idx, rel in enumerate(EDGE_RELATIONS)}
NUM_EDGE_RELATIONS = len(EDGE_RELATIONS)

_ID_LPAREN = TOKEN_TO_ID["("]
_ID_RPAREN = TOKEN_TO_ID[")"]
_ID_LBRACK = TOKEN_TO_ID["["]
_ID_RBRACK = TOKEN_TO_ID["]"]
_ID_LBRACE = TOKEN_TO_ID["{"]
_ID_RBRACE = TOKEN_TO_ID["}"]
_ID_SUB = TOKEN_TO_ID["_"]
_ID_SUP = TOKEN_TO_ID["^"]

# 兼容 MathWriting LaTeX 常见等价命令归一化
COMMAND_ALIASES: Dict[str, str] = {
    "\\mid": "|",
    "\\vert": "|",
    "\\lvert": "|",
    "\\rvert": "|",
    "\\longrightarrow": "\\to",
    "\\rightarrow": "\\to",
    "\\implies": "\\Rightarrow",
    "\\leq": "\\le",
    "\\leqslant": "\\le",
    "\\geq": "\\ge",
    "\\geqslant": "\\ge",
    "\\ne": "\\neq",
    "\\epsilon": "\\varepsilon",
    "\\emptyset": "\\varnothing",
    "\\overline": "\\bar",
    "\\widehat": "\\hat",
    "\\dfrac": "\\frac",
    "\\tfrac": "\\frac",
    "\\pmatrix": "\\matrix",
    "\\bmatrix": "\\matrix",
}

IGNORED_COMMANDS = {
    "\\left", "\\right", "\\displaystyle", "\\textstyle", "\\mathrm", "\\mathbf", "\\mathit",
    "\\text", "\\operatorname", "\\limits", "\\quad", "\\qquad", "\\,", "\\;", "\\!", "\\:",
    "\\begin", "\\end",
}

_LATEX_TOKEN_RE = re.compile(r"(\\[A-Za-z]+|[0-9]|[A-Za-z]|[^\s])")


def tokenize_latex_to_ids(latex: str) -> List[int]:
    """将标准 MathWriting / 大学数学 LaTeX 字符串清洗并映射为 CTC Token ID 序列。"""
    cleaned = latex.strip()
    cleaned = re.sub(r"\\begin\{(?:p|b|v|V)?matrix\}", r"\\matrix", cleaned)
    cleaned = re.sub(r"\\end\{(?:p|b|v|V)?matrix\}", "", cleaned)

    ids: List[int] = []
    for raw in _LATEX_TOKEN_RE.findall(cleaned):
        if raw in IGNORED_COMMANDS:
            continue
        tok = COMMAND_ALIASES.get(raw, raw)
        if tok in TOKEN_TO_ID:
            ids.append(TOKEN_TO_ID[tok])
        elif tok.startswith("\\") and len(tok) > 1:
            for ch in tok[1:]:
                if ch in TOKEN_TO_ID:
                    ids.append(TOKEN_TO_ID[ch])
    return ids


def format_tokens_to_latex(tokens: List[str]) -> str:
    """将解码后的 Token 列表拼接为规范可读的 LaTeX 数学字符串。"""
    pieces: List[str] = []
    for tok in tokens:
        if tok.startswith("\\") or (pieces and pieces[-1].startswith("\\") and tok.isalnum()):
            pieces.append(" " + tok if pieces else tok)
        else:
            pieces.append(tok)
    return "".join(pieces).strip()


def decode_ctc_greedy(pred_ids: List[int]) -> str:
    """将模型逐帧输出的 ID 序列执行 CTC 贪婪去重与空白折叠，还原数学表达式串。"""
    out: List[str] = []
    prev = -1
    for idx in pred_ids:
        if idx != 0 and idx != prev and 0 <= idx < VOCAB_SIZE:
            out.append(MATH_VOCAB[idx])
        prev = idx
    return format_tokens_to_latex(out)


def compute_ctc_token_posteriors(logits_2d) -> np.ndarray:
    """提取 125 维符号边缘存在后验 P_ctc(v) ∈ [0, 1]^V（与 TS computeCtcTokenPosteriors 对齐）。"""
    arr = np.asarray(logits_2d, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((VOCAB_SIZE,), dtype=np.float32)
    m = arr.max(axis=-1, keepdims=True)
    probs = np.exp(arr - m)
    probs /= np.clip(probs.sum(axis=-1, keepdims=True), 1e-12, None)
    max_p = probs.max(axis=0)
    log_not = np.sum(np.log(np.clip(1.0 - probs * 0.65, 1e-6, 1.0)), axis=0)
    noisy_or = 1.0 - np.exp(log_not)
    post = np.clip(0.55 * max_p + 0.45 * noisy_or, 0.0, 1.0)
    post[0] = 0.0
    return post.astype(np.float32)


def _log_add_exp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _grammar_transition_penalty(prev_ids: Sequence[int], next_id: int) -> float:
    prev_id = prev_ids[-1] if prev_ids else -1
    if (next_id in (_ID_SUP, _ID_SUB)) and (prev_id in (_ID_SUP, _ID_SUB)):
        return -3.2
    if not prev_ids and next_id in (_ID_SUP, _ID_SUB, _ID_RPAREN, _ID_RBRACK, _ID_RBRACE):
        return -3.5
    if next_id in (_ID_RPAREN, _ID_RBRACK, _ID_RBRACE):
        open_target = _ID_LPAREN if next_id == _ID_RPAREN else (_ID_LBRACK if next_id == _ID_RBRACK else _ID_LBRACE)
        depth = 0
        for pid in prev_ids:
            if pid == open_target:
                depth += 1
            elif pid == next_id:
                depth -= 1
        if depth <= 0:
            return -3.4
        if prev_id == open_target:
            return -2.4
    return 0.0


def _unclosed_delimiters(ids: Sequence[int]) -> Tuple[int, int, int, bool]:
    b, p, k = 0, 0, 0
    for pid in ids:
        if pid == _ID_LBRACE:
            b = max(0, b + 1)
        elif pid == _ID_RBRACE:
            b = max(0, b - 1)
        elif pid == _ID_LPAREN:
            p = max(0, p + 1)
        elif pid == _ID_RPAREN:
            p = max(0, p - 1)
        elif pid == _ID_LBRACK:
            k = max(0, k + 1)
        elif pid == _ID_RBRACK:
            k = max(0, k - 1)
    dangling = bool(ids and ids[-1] in (_ID_SUP, _ID_SUB))
    return b, p, k, dangling


def _repair_balanced_tokens(ids: Sequence[int]) -> List[int]:
    out = list(ids)
    while out and out[-1] in (_ID_SUP, _ID_SUB):
        out.pop()
    b, p, k, _ = _unclosed_delimiters(out)
    out.extend([_ID_RBRACE] * min(3, b))
    out.extend([_ID_RPAREN] * min(2, p))
    out.extend([_ID_RBRACK] * min(2, k))
    return out


def decode_ctc_beam_search(
    logits_2d,
    beam_width: int = 4,
    top_k: int = 6,
    prior_bias: Optional[np.ndarray] = None,
    enforce_grammar: bool = False,
) -> str:
    """
    轻量 CTC Prefix Beam Search 解码（严格对齐移动端 TypeScript `decodeCtcBeamSearch`）：
    可选注入题谱词表先验浅融合 `prior_bias` 与 TAMER 语法栈闭合约束 `enforce_grammar`。
    """
    arr = np.asarray(logits_2d, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return ""
    if prior_bias is not None and len(prior_bias) == arr.shape[1]:
        adj = arr.copy()
        adj[:, 1:] += np.asarray(prior_bias[1:], dtype=np.float32)[None, :]
    else:
        adj = arr

    m = adj.max(axis=-1, keepdims=True)
    log_probs = adj - m - np.log(np.exp(adj - m).sum(axis=-1, keepdims=True))

    neg_inf = -math.inf
    beams: Dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, neg_inf)}

    t_len, v_size = log_probs.shape
    k_eff = min(top_k, v_size)

    for t in range(t_len):
        frame = log_probs[t]
        top_indices = np.argpartition(-frame, k_eff - 1)[:k_eff].tolist()
        if 0 not in top_indices:
            top_indices.append(0)

        next_beams: Dict[tuple[int, ...], tuple[float, float]] = {}
        for prefix, (pb, pnb) in beams.items():
            p_tot = _log_add_exp(pb, pnb)
            last_tok = prefix[-1] if prefix else -1

            for idx in top_indices:
                lp = float(frame[idx])
                if idx == 0:
                    nb_b, nb_nb = next_beams.get(prefix, (neg_inf, neg_inf))
                    next_beams[prefix] = (_log_add_exp(nb_b, p_tot + lp), nb_nb)
                elif idx == last_tok:
                    nb_b, nb_nb = next_beams.get(prefix, (neg_inf, neg_inf))
                    next_beams[prefix] = (nb_b, _log_add_exp(nb_nb, pnb + lp))
                    g_pen = _grammar_transition_penalty(prefix, idx) if enforce_grammar else 0.0
                    ext = prefix + (idx,)
                    eb_b, eb_nb = next_beams.get(ext, (neg_inf, neg_inf))
                    next_beams[ext] = (eb_b, _log_add_exp(eb_nb, pb + lp + g_pen))
                else:
                    g_pen = _grammar_transition_penalty(prefix, idx) if enforce_grammar else 0.0
                    ext = prefix + (idx,)
                    eb_b, eb_nb = next_beams.get(ext, (neg_inf, neg_inf))
                    next_beams[ext] = (eb_b, _log_add_exp(eb_nb, p_tot + lp + g_pen))

        ranked = sorted(
            next_beams.items(),
            key=lambda kv: _log_add_exp(kv[1][0], kv[1][1]),
            reverse=True,
        )[:beam_width]
        beams = dict(ranked)

    if not beams:
        return ""

    def final_score(kv: Tuple[Tuple[int, ...], Tuple[float, float]]) -> float:
        pref, (pb, pnb) = kv
        sc = _log_add_exp(pb, pnb)
        if enforce_grammar:
            b, p, k, dangling = _unclosed_delimiters(pref)
            sc -= (b + p + k) * 1.15 + (1.8 if dangling else 0.0)
        return sc

    best_prefix = max(beams.items(), key=final_score)[0]
    final_ids = _repair_balanced_tokens(best_prefix) if enforce_grammar else list(best_prefix)
    tokens = [MATH_VOCAB[i] for i in final_ids if 0 < i < VOCAB_SIZE]
    return format_tokens_to_latex(tokens)
