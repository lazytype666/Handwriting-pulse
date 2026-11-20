"""
《笔迹脉搏》Google MathWriting InkML 解析与等弧长 6 通道空间重采样管线。

核心规范（与移动端 `src/engine/neuralRuntime.ts` 严格逐行一致）：
1. 彻底丢弃原始时间戳 t，只取 (x, y) 空间坐标；
2. 对每一笔按归一化字符高度后的等弧长步长 (step_norm = 0.08) 执行线性重采样，消除 60Hz/120Hz 与快慢笔速差异；
3. 输出 6 通道空间特征：[dx, dy, cos_theta, sin_theta, pen_state, y_rel]。
"""

from __future__ import annotations

import io
import math
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vocab import tokenize_latex_to_ids

SPATIAL_DIM = 6
DEFAULT_STEP_NORM = 0.085
MAX_SPATIAL_LEN = 192


def arc_length_resample_strokes(
    strokes_xy: Sequence[Sequence[Tuple[float, float]]],
    step_norm: float = DEFAULT_STEP_NORM,
    max_len: int = MAX_SPATIAL_LEN,
) -> np.ndarray:
    """
    将变长原始笔画坐标重采样为时间解耦的 6 通道空间张量 [L, 6]:
      [dx, dy, cos_theta, sin_theta, pen_state, y_rel]
    采用“零截断自适应弧长步长”：短式保持 0.085 高精度步长，长式自动放大步长保证整条公式 100% 完整装入 max_len。
    """
    valid_strokes = [list(s) for s in strokes_xy if len(s) >= 1]
    if not valid_strokes:
        return np.zeros((1, SPATIAL_DIM), dtype=np.float32)

    all_x = [pt[0] for s in valid_strokes for pt in s]
    all_y = [pt[1] for s in valid_strokes for pt in s]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    scale = max(12.0, max_y - min_y, 0.25 * (max_x - min_x))
    center_y = 0.5 * (min_y + max_y)

    # 预计算各笔归一化弧长，动态调整 eff_step 杜绝长公式尾部截断
    norm_strokes: List[Tuple[List[Tuple[float, float]], List[float]]] = []
    total_arc_all = 0.0
    for s in valid_strokes:
        norm_pts = [(pt[0] / scale, pt[1] / scale) for pt in s]
        cum_len = [0.0]
        for i in range(1, len(norm_pts)):
            d = math.hypot(norm_pts[i][0] - norm_pts[i - 1][0], norm_pts[i][1] - norm_pts[i - 1][1])
            cum_len.append(cum_len[-1] + d)
        total_arc_all += cum_len[-1]
        norm_strokes.append((norm_pts, cum_len))

    budget = max(8, max_len - len(norm_strokes) - 2)
    eff_step = max(step_norm, total_arc_all / budget) if total_arc_all > 1e-4 else step_norm

    resampled_points: List[Tuple[float, float, float]] = []  # (x_norm, y_norm, pen_state_after)

    for norm_pts, cum_len in norm_strokes:
        if len(norm_pts) == 1:
            resampled_points.append((norm_pts[0][0], norm_pts[0][1], 0.0))
            continue

        total_len = cum_len[-1]
        if total_len < 1e-4:
            resampled_points.append((norm_pts[0][0], norm_pts[0][1], 0.0))
            continue

        n_steps = max(2, min(64, int(math.ceil(total_len / eff_step)) + 1))
        targets = np.linspace(0.0, total_len, num=n_steps)
        seg_idx = 0
        for idx_t, target_s in enumerate(targets):
            while seg_idx + 1 < len(cum_len) and cum_len[seg_idx + 1] < target_s:
                seg_idx += 1
            s0 = cum_len[seg_idx]
            s1 = cum_len[min(seg_idx + 1, len(cum_len) - 1)]
            p0 = norm_pts[seg_idx]
            p1 = norm_pts[min(seg_idx + 1, len(norm_pts) - 1)]
            alpha = (target_s - s0) / max(1e-6, s1 - s0) if s1 > s0 else 0.0
            rx = p0[0] + alpha * (p1[0] - p0[0])
            ry = p0[1] + alpha * (p1[1] - p0[1])
            pen_state = 0.0 if idx_t == len(targets) - 1 else 1.0
            resampled_points.append((rx, ry, pen_state))

        if len(resampled_points) >= max_len:
            resampled_points = resampled_points[:max_len]
            break

    if not resampled_points:
        return np.zeros((1, SPATIAL_DIM), dtype=np.float32)

    out = np.zeros((len(resampled_points), SPATIAL_DIM), dtype=np.float32)
    origin_x = resampled_points[0][0]
    norm_center_y = center_y / scale

    for i, (px, py, pen_s) in enumerate(resampled_points):
        if i == 0:
            dx, dy = 0.0, 0.0
        else:
            prev_x, prev_y, _ = resampled_points[i - 1]
            dx = float(max(-4.0, min(4.0, px - prev_x)))
            dy = float(max(-4.0, min(4.0, py - prev_y)))
        ds = math.hypot(dx, dy)
        cos_t = dx / ds if ds > 1e-6 else 1.0
        sin_t = dy / ds if ds > 1e-6 else 0.0
        y_rel = float(max(-1.5, min(1.5, py - norm_center_y)))
        out[i] = [dx, dy, cos_t, sin_t, pen_s, y_rel]

    return out


def parse_inkml_bytes(raw_bytes: bytes) -> Optional[Tuple[List[List[Tuple[float, float]]], str]]:
    """解析单个 Google MathWriting InkML 文件，返回 (strokes_xy, latex_label)。"""
    try:
        root = ET.fromstring(raw_bytes)
    except ET.ParseError:
        return None

    # 剥离命名空间便于统一查找
    ns_prefix = ""
    if root.tag.startswith("{"):
        ns_prefix = root.tag[: root.tag.index("}") + 1]

    label = ""
    fallback_label = ""
    for ann in root.findall(f".//{ns_prefix}annotation"):
        ann_type = ann.attrib.get("type", "")
        text = (ann.text or "").strip()
        if ann_type == "normalizedLabel" and text:
            label = text
            break
        if ann_type in ("truth", "label") and text and not fallback_label:
            fallback_label = text

    final_label = label or fallback_label
    if not final_label:
        return None

    strokes: List[List[Tuple[float, float]]] = []
    for trace in root.findall(f".//{ns_prefix}trace"):
        raw_trace = (trace.text or "").strip()
        if not raw_trace:
            continue
        pts: List[Tuple[float, float]] = []
        for token in raw_trace.split(","):
            parts = token.strip().split()
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if pts:
            strokes.append(pts)

    if not strokes:
        return None
    return strokes, final_label


def _process_inkml_record(
    raw_bytes: bytes, max_seq_len: int
) -> Optional[Tuple[np.ndarray, List[int], str]]:
    parsed = parse_inkml_bytes(raw_bytes)
    if parsed is None:
        return None
    strokes_xy, latex = parsed
    token_ids = tokenize_latex_to_ids(latex)
    if not token_ids or len(token_ids) > 40:
        return None
    feat = arc_length_resample_strokes(strokes_xy, max_len=max_seq_len)
    if len(feat) < 4 or len(feat) < len(token_ids) * 2:
        return None
    return feat, token_ids, latex


def _process_inkml_file(args: Tuple[str, int]) -> Optional[Tuple[np.ndarray, List[int], str]]:
    path_str, max_seq_len = args
    try:
        return _process_inkml_record(Path(path_str).read_bytes(), max_seq_len=max_seq_len)
    except OSError:
        return None


def load_mathwriting_subset_from_dir(
    mw_root: str | Path,
    splits: Sequence[str] = ("symbols", "train"),
    max_samples: int = 30000,
    max_seq_len: int = MAX_SPATIAL_LEN,
    shuffle_seed: Optional[int] = 42,
    num_workers: int = 8,
) -> List[Tuple[np.ndarray, List[int], str]]:
    """高速并行从已解压的 MathWriting 目录读取 `.inkml` 样本（均匀打乱抽样 + 多进程解析）。"""
    from concurrent.futures import ProcessPoolExecutor

    root_path = Path(mw_root)
    if (root_path / "mathwriting-2024").is_dir():
        root_path = root_path / "mathwriting-2024"

    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    candidate_paths: List[str] = []
    remaining = max_samples
    for idx_sp, sp in enumerate(splits):
        sp_dir = root_path / sp
        if not sp_dir.is_dir() or remaining <= 0:
            continue
        files = sorted(sp_dir.glob("*.inkml"))
        if rng is not None and len(files) > 1:
            perm = rng.permutation(len(files))
            files = [files[i] for i in perm]
        # 优先收全 symbols，剩余额度全部留给后续 split（例如 train），并多取 15% 冗余抵消非法样本过滤
        is_last = idx_sp == len(splits) - 1
        take = len(files) if (sp == "symbols" and len(files) < remaining) else (
            int(remaining * 1.15) + 64 if is_last else min(len(files), remaining)
        )
        chosen = files[: min(len(files), take)]
        candidate_paths.extend(str(p) for p in chosen)
        if sp == "symbols":
            remaining = max(0, remaining - len(chosen))

    if not candidate_paths:
        return []

    results: List[Tuple[np.ndarray, List[int], str]] = []
    tasks = [(p, max_seq_len) for p in candidate_paths]
    if num_workers > 1 and len(tasks) >= 500:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for item in pool.map(_process_inkml_file, tasks, chunksize=128):
                if item is not None:
                    results.append(item)
                    if len(results) >= max_samples:
                        break
    else:
        for t in tasks:
            item = _process_inkml_file(t)
            if item is not None:
                results.append(item)
                if len(results) >= max_samples:
                    break
    return results


def load_mathwriting_subset_from_tgz(
    tgz_path: str | Path,
    split_prefix: str = "train/",
    max_samples: int = 25000,
    max_seq_len: int = MAX_SPATIAL_LEN,
) -> List[Tuple[np.ndarray, List[int], str]]:
    """流式从 `mathwriting-2024.tgz` 解压加载指定子集并转为 6 通道重采样张量。"""
    results: List[Tuple[np.ndarray, List[int], str]] = []
    with tarfile.open(tgz_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".inkml"):
                continue
            if split_prefix and split_prefix not in member.name:
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            item = _process_inkml_record(fobj.read(), max_seq_len=max_seq_len)
            if item is None:
                continue
            results.append(item)
            if len(results) >= max_samples:
                break
    return results


# ---------------------------- 大学数学核心符号随机笔迹增强生成器 ----------------------------

def _prototype_glyph(sym: str, ox: float, oy: float, h: float, rng: np.random.Generator) -> List[List[Tuple[float, float]]]:
    """按手写几何规范生成带自然抖动与斜体变形的大学数学单符号多笔画轨迹。"""
    w = h * rng.uniform(0.55, 0.85)
    t_vals = np.linspace(0, 1, 10)
    tilt = rng.uniform(-0.12, 0.12)

    def warp(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [
            (
                ox + x + tilt * (y - h * 0.5) + float(rng.normal(0, 0.025 * h)),
                oy + y + float(rng.normal(0, 0.025 * h)),
            )
            for x, y in pts
        ]

    if sym == "=":
        y1, y2 = h * 0.36, h * 0.66
        return [
            warp([(w * 0.05, y1), (w * 0.95, y1)]),
            warp([(w * 0.05, y2), (w * 0.95, y2)]),
        ]
    if sym == "+":
        return [
            warp([(w * 0.5, h * 0.15), (w * 0.5, h * 0.85)]),
            warp([(w * 0.1, h * 0.5), (w * 0.9, h * 0.5)]),
        ]
    if sym == "-":
        return [warp([(w * 0.1, h * 0.5), (w * 0.9, h * 0.5)])]
    if sym == "\\frac":
        fw = w * 1.4
        return [warp([(0.0, h * 0.5), (fw, h * 0.5)])]
    if sym == "|":
        return [warp([(w * 0.5, -h * 0.25), (w * 0.5, h * 1.25)])]
    if sym == "\\int":
        pts = [
            (w * 0.72, -h * 0.28),
            (w * 0.52, -h * 0.32),
            (w * 0.48, h * 0.0),
            (w * 0.46, h * 0.5),
            (w * 0.44, h * 1.0),
            (w * 0.36, h * 1.28),
            (w * 0.16, h * 1.24),
        ]
        return [warp(pts)]
    if sym == "\\sum":
        pts = [
            (w * 0.9, h * 0.08),
            (w * 0.1, h * 0.08),
            (w * 0.62, h * 0.5),
            (w * 0.1, h * 0.92),
            (w * 0.92, h * 0.92),
        ]
        return [warp(pts)]
    if sym == "\\to":
        return [
            warp([(0.0, h * 0.5), (w * 0.95, h * 0.5)]),
            warp([(w * 0.68, h * 0.28), (w * 0.95, h * 0.5), (w * 0.68, h * 0.72)]),
        ]
    if sym == "~":
        pts = [(t * w, h * 0.5 - math.sin(t * 2 * math.pi) * h * 0.16) for t in t_vals]
        return [warp(pts)]
    if sym == "\\infty":
        pts = [
            (w * 0.5 + math.sin(t * 2 * math.pi) * w * 0.45, h * 0.5 + math.sin(t * 4 * math.pi) * h * 0.24)
            for t in np.linspace(0, 1, 16)
        ]
        return [warp(pts)]
    if sym == "\\Phi":
        ring = [
            (w * 0.5 + math.cos(t * 2 * math.pi) * w * 0.38, h * 0.5 + math.sin(t * 2 * math.pi) * h * 0.26)
            for t in np.linspace(0, 1, 14)
        ]
        bar = [(w * 0.5, h * 0.05), (w * 0.5, h * 0.95)]
        return [warp(ring), warp(bar)]
    if sym == "0":
        ring = [
            (w * 0.5 + math.cos(t * 2 * math.pi) * w * 0.38, h * 0.5 + math.sin(t * 2 * math.pi) * h * 0.42)
            for t in np.linspace(0, 1, 12)
        ]
        return [warp(ring)]
    if sym == "1":
        return [warp([(w * 0.48, h * 0.1), (w * 0.5, h * 0.9)])]
    if sym == "x":
        return [
            warp([(w * 0.15, h * 0.25), (w * 0.85, h * 0.85)]),
            warp([(w * 0.85, h * 0.25), (w * 0.15, h * 0.85)]),
        ]
    if sym == "\\lambda":
        return [
            warp([(w * 0.25, h * 0.1), (w * 0.82, h * 0.92)]),
            warp([(w * 0.52, h * 0.48), (w * 0.15, h * 0.92)]),
        ]
    if sym == "(":
        pts = [(w * (0.68 - 0.32 * math.sin(t * math.pi)), h * (0.05 + 0.9 * t)) for t in t_vals]
        return [warp(pts)]
    if sym == ")":
        pts = [(w * (0.32 + 0.32 * math.sin(t * math.pi)), h * (0.05 + 0.9 * t)) for t in t_vals]
        return [warp(pts)]
    if sym == "A":
        return [
            warp([(w * 0.5, h * 0.08), (w * 0.12, h * 0.92)]),
            warp([(w * 0.5, h * 0.08), (w * 0.88, h * 0.92)]),
            warp([(w * 0.26, h * 0.6), (w * 0.74, h * 0.6)]),
        ]
    if sym == "P":
        return [
            warp([(w * 0.22, h * 0.08), (w * 0.22, h * 0.92)]),
            warp([(w * 0.22, h * 0.08), (w * 0.76, h * 0.16), (w * 0.76, h * 0.46), (w * 0.22, h * 0.52)]),
        ]
    if sym == "e":
        pts = [
            (w * 0.18, h * 0.52),
            (w * 0.8, h * 0.48),
            (w * 0.72, h * 0.28),
            (w * 0.32, h * 0.28),
            (w * 0.18, h * 0.55),
            (w * 0.36, h * 0.82),
            (w * 0.82, h * 0.78),
        ]
        return [warp(pts)]
    # 兜底随机多折字母骨架
    pts = [(w * (0.2 + 0.6 * ((i % 2))), h * (0.15 + 0.7 * (i / 3))) for i in range(4)]
    return [warp(pts)]


UNIVERSITY_TEMPLATES: List[Tuple[List[str], str]] = [
    (["\\int", "x", "+", "1", "=", "0"], "\\int x + 1 = 0"),
    (["\\sum", "x", "=", "1"], "\\sum x = 1"),
    (["x", "\\to", "0"], "x \\to 0"),
    (["|", "A", "-", "\\lambda", "e", "|", "=", "0"], "|A - \\lambda e| = 0"),
    (["P", "(", "A", "|", "e", ")", "=", "0"], "P(A|e) = 0"),
    (["x", "~", "\\Phi", "(", "0", ")"], "x ~ \\Phi(0)"),
    (["A", "=", "\\infty"], "A = \\infty"),
    (["\\int", "e", "-", "1", "=", "0"], "\\int e - 1 = 0"),
    (["|", "A", "|", "=", "0"], "|A| = 0"),
    (["1", "+", "x", "=", "\\infty"], "1 + x = \\infty"),
    (["\\lambda", "-", "1", "=", "0"], "\\lambda - 1 = 0"),
    (["\\sum", "1", "+", "0", "=", "1"], "\\sum 1 + 0 = 1"),
]


def generate_university_math_sequences(
    n_samples: int = 2000,
    seed: int = 2026,
) -> List[Tuple[np.ndarray, List[int], str]]:
    """生成带自然几何抖动的大学数学表达样本（配合真实 MathWriting 数据集联合强化大学数学专有符号）。"""
    rng = np.random.default_rng(seed)
    outputs: List[Tuple[np.ndarray, List[int], str]] = []
    for i in range(n_samples):
        syms, latex = UNIVERSITY_TEMPLATES[i % len(UNIVERSITY_TEMPLATES)]
        h = float(rng.uniform(28.0, 56.0))
        ox = float(rng.uniform(20.0, 60.0))
        oy = float(rng.uniform(40.0, 120.0))
        all_strokes: List[List[Tuple[float, float]]] = []
        for s in syms:
            glyph_strokes = _prototype_glyph(s, ox, oy + float(rng.normal(0, 2.0)), h, rng)
            all_strokes.extend(glyph_strokes)
            ox += h * rng.uniform(0.75, 1.05)

        feat = arc_length_resample_strokes(all_strokes, max_len=MAX_SPATIAL_LEN)
        token_ids = tokenize_latex_to_ids(latex)
        if token_ids and len(feat) >= len(token_ids) * 2:
            outputs.append((feat, token_ids, latex))
    return outputs
