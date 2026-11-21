"""
《笔迹脉搏》全大学数学 CTC 符号词表（高数 + 线性代数 + 概率论与数理统计）。

词表索引 0 固定保留为 `<blank>`（CTC 解码空白符），共 120 个 Token。
与移动端 `src/engine/mathVocab.ts` 保持严格 1-to-1 索引一致，保证 PyTorch/ONNX 与端侧 Float32Array 解码零偏差。
"""

from __future__ import annotations

import re
from typing import Dict, List

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
    # 剥离环境名如 \begin{pmatrix} -> \matrix
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
            # 未知文本类命令按字母拆入（如 \sin -> s,i,n; \cos -> c,o,s; \ln -> l,n）
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


def _log_add_exp(a: float, b: float) -> float:
    import math

    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def decode_ctc_beam_search(
    logits_2d,
    beam_width: int = 4,
    top_k: int = 6,
) -> str:
    """
    轻量 CTC Prefix Beam Search 解码（严格对齐移动端 TypeScript `decodeCtcBeamSearch`）：
    在每帧仅展开 top_k 高概率候选符号，合并同一前缀的 blank / non-blank 路径概率，显著减少拼写残缺。
    """
    import math
    import numpy as np

    arr = np.asarray(logits_2d, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return ""

    # 逐帧 log_softmax
    m = arr.max(axis=-1, keepdims=True)
    log_probs = arr - m - np.log(np.exp(arr - m).sum(axis=-1, keepdims=True))

    neg_inf = -math.inf
    # beams: prefix_tuple -> (p_blank, p_non_blank)
    beams: Dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, neg_inf)}

    t_len, v_size = log_probs.shape
    k_eff = min(top_k, v_size)

    for t in range(t_len):
        frame = log_probs[t]
        # 始终保留 blank(0) 以及概率最高的 top_k 符号
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
                    # 连续同符号：1) 从 non-blank 继续停留不新增 token；2) 从 blank 跨越则新增重复 token
                    nb_b, nb_nb = next_beams.get(prefix, (neg_inf, neg_inf))
                    next_beams[prefix] = (nb_b, _log_add_exp(nb_nb, pnb + lp))
                    ext = prefix + (idx,)
                    eb_b, eb_nb = next_beams.get(ext, (neg_inf, neg_inf))
                    next_beams[ext] = (eb_b, _log_add_exp(eb_nb, pb + lp))
                else:
                    ext = prefix + (idx,)
                    eb_b, eb_nb = next_beams.get(ext, (neg_inf, neg_inf))
                    next_beams[ext] = (eb_b, _log_add_exp(eb_nb, p_tot + lp))

        # 剪枝保留 top beam_width 条前缀
        ranked = sorted(
            next_beams.items(),
            key=lambda kv: _log_add_exp(kv[1][0], kv[1][1]),
            reverse=True,
        )[:beam_width]
        beams = dict(ranked)

    if not beams:
        return ""
    best_prefix = max(beams.items(), key=lambda kv: _log_add_exp(kv[1][0], kv[1][1]))[0]
    tokens = [MATH_VOCAB[i] for i in best_prefix if 0 < i < VOCAB_SIZE]
    return format_tokens_to_latex(tokens)
