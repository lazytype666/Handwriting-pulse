"""
《笔迹脉搏》Google MathWriting InkML 解析、二维拓扑规范化重排、等弧长 10 通道空间重采样与 2D 笔画图样本生成器。

核心规范（与移动端 `src/engine/neuralRuntime.ts` 严格逐维一致）：
1. 彻底丢弃原始时间戳 t，只取 (x, y) 空间坐标；
2. 对每一笔按归一化字符高度后的等弧长步长 (step_norm = 0.085) 执行零截断自适应线性重采样；
3. 输出 10 通道空间特征：
   [dx, dy, cos_theta, sin_theta, pen_state, y_rel, st_cx_rel, st_cy_rel, st_w_norm, st_h_norm]
   （后 4 维编码每点所属笔画的表达式相对包围盒中心与归一化尺度，支撑上/下标、分式线与极限边界判别）；
4. 依托 MathWriting `symbols/` 真实人类手写符号库构造含真值 `SameSymbol / Right / Above / Below / Sup / Sub` 边的 2D 大学数学图样本，供训练双线性笔画图分类头。
"""

from __future__ import annotations

import math
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vocab import EDGE_REL_TO_ID, tokenize_latex_to_ids

SPATIAL_DIM = 10
EDGE_GEOM_DIM = 10
DEFAULT_STEP_NORM = 0.085
MAX_SPATIAL_LEN = 192


def _stroke_bbox(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = max(1e-3, x1 - x0)
    h = max(1e-3, y1 - y0)
    return x0, y0, w, h, x0 + 0.5 * w, y0 + 0.5 * h


def canonicalize_strokes_xy(
    strokes_xy: Sequence[Sequence[Tuple[float, float]]],
) -> List[List[Tuple[float, float]]]:
    """
    二维拓扑规范化重排（Spatial Canonical Stroke Reordering）：
    在送入 Conformer-CTC 前解除“回头补写分子、补写积分上限或局部修订”造成的纯时间乱序：
    1) 检测水平真分式线，将分式按 [分子笔画] -> [分式线] -> [分母笔画] 归组排定；
    2) 检测高大竖直微积分主算符（如 ∫/∑），将贴靠其上下端的界限簇排序至主体之后；
    3) 其余同级笔画按局部横向中心顺序排列（跨距极近的重叠多笔画保持稳定序）。
    """
    valid = [list(s) for s in strokes_xy if len(s) >= 1]
    if len(valid) <= 2:
        return valid

    meta = []
    heights = []
    for idx, s in enumerate(valid):
        x0, y0, w, h, cx, cy = _stroke_bbox(s)
        dx_net = abs(s[-1][0] - s[0][0])
        dy_net = abs(s[-1][1] - s[0][1])
        horiz = dx_net / max(1.0, dx_net + dy_net)
        vert = dy_net / max(1.0, dx_net + dy_net)
        meta.append({
            "idx": idx,
            "pts": s,
            "x0": x0,
            "y0": y0,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "horiz": horiz,
            "vert": vert,
        })
        if 4.0 < h < 220.0:
            heights.append(h)

    avg_h = max(14.0, float(np.median(heights)) if heights else 36.0)
    used = set()
    ordered_groups: List[Tuple[float, List[int]]] = []

    # 1. 检测宽分式线及其上下支撑簇
    for m in meta:
        if m["horiz"] > 0.82 and m["w"] >= avg_h * 0.85 and m["h"] <= max(18.0, m["w"] * 0.35):
            nums = []
            dens = []
            bx0, bx1, bcy = m["x0"], m["x0"] + m["w"], m["cy"]
            for o in meta:
                if o["idx"] == m["idx"] or o["idx"] in used:
                    continue
                ox0, ox1 = o["x0"], o["x0"] + o["w"]
                inter = max(0.0, min(bx1, ox1) - max(bx0, ox0))
                x_ov = inter / max(1.0, min(m["w"], o["w"]))
                if x_ov < 0.2:
                    continue
                if 3.0 <= bcy - o["cy"] <= avg_h * 1.75:
                    nums.append(o)
                elif 3.0 <= o["cy"] - bcy <= avg_h * 1.75:
                    dens.append(o)
            if nums and dens:
                used.add(m["idx"])
                for o in nums:
                    used.add(o["idx"])
                for o in dens:
                    used.add(o["idx"])
                nums.sort(key=lambda k: (k["cx"], k["idx"]))
                dens.sort(key=lambda k: (k["cx"], k["idx"]))
                grp = [k["idx"] for k in nums] + [m["idx"]] + [k["idx"] for k in dens]
                ordered_groups.append((m["cx"], grp))

    # 2. 检测高竖 ∫/∑ 等微积分锚点与上下限簇
    for m in meta:
        if m["idx"] in used:
            continue
        if m["h"] >= avg_h * 1.3 and m["vert"] > 0.55 and m["w"] <= m["h"] * 0.75:
            lows = []
            highs = []
            for o in meta:
                if o["idx"] == m["idx"] or o["idx"] in used:
                    continue
                if o["h"] < avg_h * 0.95 and abs(o["cx"] - (m["cx"] + m["w"] * 0.35)) <= avg_h * 1.2:
                    if o["cy"] >= m["cy"] + m["h"] * 0.2:
                        lows.append(o)
                    elif o["cy"] <= m["cy"] - m["h"] * 0.2:
                        highs.append(o)
            if lows or highs:
                used.add(m["idx"])
                for o in lows:
                    used.add(o["idx"])
                for o in highs:
                    used.add(o["idx"])
                lows.sort(key=lambda k: (k["cx"], k["idx"]))
                highs.sort(key=lambda k: (k["cx"], k["idx"]))
                grp = [m["idx"]] + [k["idx"] for k in lows] + [k["idx"] for k in highs]
                ordered_groups.append((m["cx"], grp))

    for m in meta:
        if m["idx"] not in used:
            ordered_groups.append((m["cx"], [m["idx"]]))

    ordered_groups.sort(key=lambda item: (item[0], item[1][0]))
    out: List[List[Tuple[float, float]]] = []
    for _, idx_list in ordered_groups:
        for i in idx_list:
            out.append(valid[i])
    return out


def extract_pair_geom_10(
    pts_a: Sequence[Tuple[float, float]],
    pts_b: Sequence[Tuple[float, float]],
    expr_scale: float,
) -> np.ndarray:
    """提取任意两笔的 10 维尺度不变二维几何关系特征（与 TS NeuralRuntime 逐维一致）。"""
    s = max(12.0, float(expr_scale))
    ax0, ay0, aw, ah, acx, acy = _stroke_bbox(pts_a)
    bx0, by0, bw, bh, bcx, bcy = _stroke_bbox(pts_b)
    inter_x = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    inter_y = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    x_ov = inter_x / max(1e-3, min(aw, bw))
    y_ov = inter_y / max(1e-3, min(ah, bh))

    gap_x = max(0.0, bx0 - (ax0 + aw), ax0 - (bx0 + bw))
    gap_y = max(0.0, by0 - (ay0 + ah), ay0 - (by0 + bh))
    min_dist = math.hypot(gap_x, gap_y) / s
    area_a = max(1.0, aw * ah)
    area_b = max(1.0, bw * bh)
    log_area = max(-3.0, min(3.0, math.log(area_b / area_a)))

    return np.asarray(
        [
            max(-3.0, min(3.0, (bcx - acx) / s)),
            max(-3.0, min(3.0, (bcy - acy) / s)),
            max(0.0, min(3.0, aw / s)),
            max(0.0, min(3.0, ah / s)),
            max(0.0, min(3.0, bw / s)),
            max(0.0, min(3.0, bh / s)),
            max(0.0, min(1.0, x_ov)),
            max(0.0, min(1.0, y_ov)),
            max(0.0, min(3.0, min_dist)),
            log_area,
        ],
        dtype=np.float32,
    )


def arc_length_resample_strokes(
    strokes_xy: Sequence[Sequence[Tuple[float, float]]],
    step_norm: float = DEFAULT_STEP_NORM,
    max_len: int = MAX_SPATIAL_LEN,
    spatial_dim: int = SPATIAL_DIM,
) -> np.ndarray:
    """
    将变长原始笔画坐标重采样为时间解耦的 10 通道空间张量 [L, 10]（或 6 通道兼容模式）：
      [dx, dy, cos_theta, sin_theta, pen_state, y_rel, st_cx_rel, st_cy_rel, st_w_norm, st_h_norm]
    采用“零截断自适应弧长步长”：短式保持 0.085 高精度步长，长式自动放大步长保证整条公式 100% 完整装入 max_len。
    """
    valid_strokes = [list(s) for s in strokes_xy if len(s) >= 1]
    if not valid_strokes:
        return np.zeros((1, spatial_dim), dtype=np.float32)

    all_x = [pt[0] for s in valid_strokes for pt in s]
    all_y = [pt[1] for s in valid_strokes for pt in s]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    scale = max(12.0, max_y - min_y, 0.25 * (max_x - min_x))
    norm_center_x = (0.5 * (min_x + max_x)) / scale
    norm_center_y = (0.5 * (min_y + max_y)) / scale

    norm_strokes: List[Tuple[List[Tuple[float, float]], List[float], Tuple[float, float, float, float]]] = []
    total_arc_all = 0.0
    for s in valid_strokes:
        norm_pts = [(pt[0] / scale, pt[1] / scale) for pt in s]
        s_xs = [p[0] for p in norm_pts]
        s_ys = [p[1] for p in norm_pts]
        sx0, sx1 = min(s_xs), max(s_xs)
        sy0, sy1 = min(s_ys), max(s_ys)
        st_cx_rel = max(-2.5, min(2.5, 0.5 * (sx0 + sx1) - norm_center_x))
        st_cy_rel = max(-1.5, min(1.5, 0.5 * (sy0 + sy1) - norm_center_y))
        st_w_norm = max(0.0, min(3.0, sx1 - sx0))
        st_h_norm = max(0.0, min(3.0, sy1 - sy0))

        cum_len = [0.0]
        for i in range(1, len(norm_pts)):
            d = math.hypot(norm_pts[i][0] - norm_pts[i - 1][0], norm_pts[i][1] - norm_pts[i - 1][1])
            cum_len.append(cum_len[-1] + d)
        total_arc_all += cum_len[-1]
        norm_strokes.append((norm_pts, cum_len, (st_cx_rel, st_cy_rel, st_w_norm, st_h_norm)))

    budget = max(8, max_len - len(norm_strokes) - 2)
    eff_step = max(step_norm, total_arc_all / budget) if total_arc_all > 1e-4 else step_norm

    resampled_points: List[Tuple[float, float, float, float, float, float, float]] = []

    for norm_pts, cum_len, (scx, scy, sw, sh) in norm_strokes:
        if len(norm_pts) == 1:
            resampled_points.append((norm_pts[0][0], norm_pts[0][1], 0.0, scx, scy, sw, sh))
            continue

        total_len = cum_len[-1]
        if total_len < 1e-4:
            resampled_points.append((norm_pts[0][0], norm_pts[0][1], 0.0, scx, scy, sw, sh))
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
            resampled_points.append((rx, ry, pen_state, scx, scy, sw, sh))

        if len(resampled_points) >= max_len:
            resampled_points = resampled_points[:max_len]
            break

    if not resampled_points:
        return np.zeros((1, spatial_dim), dtype=np.float32)

    out = np.zeros((len(resampled_points), spatial_dim), dtype=np.float32)
    for i, (px, py, pen_s, scx, scy, sw, sh) in enumerate(resampled_points):
        if i == 0:
            dx, dy = 0.0, 0.0
        else:
            prev_x, prev_y = resampled_points[i - 1][0], resampled_points[i - 1][1]
            dx = float(max(-4.0, min(4.0, px - prev_x)))
            dy = float(max(-4.0, min(4.0, py - prev_y)))
        ds = math.hypot(dx, dy)
        cos_t = dx / ds if ds > 1e-6 else 1.0
        sin_t = dy / ds if ds > 1e-6 else 0.0
        y_rel = float(max(-1.5, min(1.5, py - norm_center_y)))
        if spatial_dim >= 10:
            out[i] = [dx, dy, cos_t, sin_t, pen_s, y_rel, scx, scy, sw, sh]
        else:
            out[i, :6] = [dx, dy, cos_t, sin_t, pen_s, y_rel]

    return out


def parse_inkml_bytes(raw_bytes: bytes) -> Optional[Tuple[List[List[Tuple[float, float]]], str]]:
    """解析单个 Google MathWriting InkML 文件，返回 (strokes_xy, latex_label)。"""
    try:
        root = ET.fromstring(raw_bytes)
    except ET.ParseError:
        return None

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
    raw_bytes: bytes,
    max_seq_len: int,
    spatial_dim: int = SPATIAL_DIM,
) -> Optional[Tuple[np.ndarray, List[int], str]]:
    parsed = parse_inkml_bytes(raw_bytes)
    if parsed is None:
        return None
    strokes_xy, latex = parsed
    token_ids = tokenize_latex_to_ids(latex)
    if not token_ids or len(token_ids) > 40:
        return None
    feat = arc_length_resample_strokes(strokes_xy, max_len=max_seq_len, spatial_dim=spatial_dim)
    if len(feat) < 4 or len(feat) < len(token_ids) * 2:
        return None
    return feat, token_ids, latex


def _process_inkml_file(args: Tuple[str, int, int]) -> Optional[Tuple[np.ndarray, List[int], str]]:
    path_str, max_seq_len, spatial_dim = args
    try:
        return _process_inkml_record(Path(path_str).read_bytes(), max_seq_len=max_seq_len, spatial_dim=spatial_dim)
    except OSError:
        return None


def load_mathwriting_subset_from_dir(
    mw_root: str | Path,
    splits: Sequence[str] = ("symbols", "train"),
    max_samples: int = 30000,
    max_seq_len: int = MAX_SPATIAL_LEN,
    shuffle_seed: Optional[int] = 42,
    num_workers: int = 8,
    spatial_dim: int = SPATIAL_DIM,
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
    tasks = [(p, max_seq_len, spatial_dim) for p in candidate_paths]
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
    spatial_dim: int = SPATIAL_DIM,
) -> List[Tuple[np.ndarray, List[int], str]]:
    """流式从 `mathwriting-2024.tgz` 解压加载指定子集并转为 10 通道重采样张量。"""
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
            item = _process_inkml_record(fobj.read(), max_seq_len=max_seq_len, spatial_dim=spatial_dim)
            if item is None:
                continue
            results.append(item)
            if len(results) >= max_samples:
                break
    return results


# ---------------------------- 大学数学核心符号原型与真人类符号拼装 ----------------------------

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
        return [warp([(w * 0.05, y1), (w * 0.95, y1)]), warp([(w * 0.05, y2), (w * 0.95, y2)])]
    if sym == "+":
        return [warp([(w * 0.5, h * 0.15), (w * 0.5, h * 0.85)]), warp([(w * 0.1, h * 0.5), (w * 0.9, h * 0.5)])]
    if sym == "-":
        return [warp([(w * 0.1, h * 0.5), (w * 0.9, h * 0.5)])]
    if sym == "\\frac":
        fw = w * 1.4
        return [warp([(0.0, h * 0.5), (fw, h * 0.5)])]
    if sym == "|":
        return [warp([(w * 0.5, -h * 0.25), (w * 0.5, h * 1.25)])]
    if sym == "\\int":
        pts = [(w * 0.72, -h * 0.28), (w * 0.52, -h * 0.32), (w * 0.48, h * 0.0), (w * 0.46, h * 0.5), (w * 0.44, h * 1.0), (w * 0.36, h * 1.28), (w * 0.16, h * 1.24)]
        return [warp(pts)]
    if sym == "\\sum":
        pts = [(w * 0.9, h * 0.08), (w * 0.1, h * 0.08), (w * 0.62, h * 0.5), (w * 0.1, h * 0.92), (w * 0.92, h * 0.92)]
        return [warp(pts)]
    if sym == "\\to":
        return [warp([(0.0, h * 0.5), (w * 0.95, h * 0.5)]), warp([(w * 0.68, h * 0.28), (w * 0.95, h * 0.5), (w * 0.68, h * 0.72)])]
    if sym == "~":
        pts = [(t * w, h * 0.5 - math.sin(t * 2 * math.pi) * h * 0.16) for t in t_vals]
        return [warp(pts)]
    if sym == "\\infty":
        pts = [(w * 0.5 + math.sin(t * 2 * math.pi) * w * 0.45, h * 0.5 + math.sin(t * 4 * math.pi) * h * 0.24) for t in np.linspace(0, 1, 16)]
        return [warp(pts)]
    if sym == "\\Phi":
        ring = [(w * 0.5 + math.cos(t * 2 * math.pi) * w * 0.38, h * 0.5 + math.sin(t * 2 * math.pi) * h * 0.26) for t in np.linspace(0, 1, 14)]
        bar = [(w * 0.5, h * 0.05), (w * 0.5, h * 0.95)]
        return [warp(ring), warp(bar)]
    if sym == "0":
        return [warp([(w * 0.5 + math.cos(t * 2 * math.pi) * w * 0.38, h * 0.5 + math.sin(t * 2 * math.pi) * h * 0.42) for t in np.linspace(0, 1, 12)])]
    if sym == "1":
        return [warp([(w * 0.48, h * 0.1), (w * 0.5, h * 0.9)])]
    if sym == "2":
        return [warp([(w * 0.2, h * 0.25), (w * 0.78, h * 0.18), (w * 0.75, h * 0.5), (w * 0.2, h * 0.88), (w * 0.85, h * 0.88)])]
    if sym == "x":
        return [warp([(w * 0.15, h * 0.25), (w * 0.85, h * 0.85)]), warp([(w * 0.85, h * 0.25), (w * 0.15, h * 0.85)])]
    if sym == "\\lambda":
        return [warp([(w * 0.25, h * 0.1), (w * 0.82, h * 0.92)]), warp([(w * 0.52, h * 0.48), (w * 0.15, h * 0.92)])]
    if sym == "(":
        return [warp([(w * (0.68 - 0.32 * math.sin(t * math.pi)), h * (0.05 + 0.9 * t)) for t in t_vals])]
    if sym == ")":
        return [warp([(w * (0.32 + 0.32 * math.sin(t * math.pi)), h * (0.05 + 0.9 * t)) for t in t_vals])]
    if sym == "A":
        return [warp([(w * 0.5, h * 0.08), (w * 0.12, h * 0.92)]), warp([(w * 0.5, h * 0.08), (w * 0.88, h * 0.92)]), warp([(w * 0.26, h * 0.6), (w * 0.74, h * 0.6)])]
    if sym == "P":
        return [warp([(w * 0.22, h * 0.08), (w * 0.22, h * 0.92)]), warp([(w * 0.22, h * 0.08), (w * 0.76, h * 0.16), (w * 0.76, h * 0.46), (w * 0.22, h * 0.52)])]
    if sym == "e":
        return [warp([(w * 0.18, h * 0.52), (w * 0.8, h * 0.48), (w * 0.72, h * 0.28), (w * 0.32, h * 0.28), (w * 0.18, h * 0.55), (w * 0.36, h * 0.82), (w * 0.82, h * 0.78)])]
    pts = [(w * (0.2 + 0.6 * (i % 2)), h * (0.15 + 0.7 * (i / 3))) for i in range(4)]
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
    spatial_dim: int = SPATIAL_DIM,
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

        feat = arc_length_resample_strokes(all_strokes, max_len=MAX_SPATIAL_LEN, spatial_dim=spatial_dim)
        token_ids = tokenize_latex_to_ids(latex)
        if token_ids and len(feat) >= len(token_ids) * 2:
            outputs.append((feat, token_ids, latex))
    return outputs


# ---------------------------- 2D 笔画图样本生成器（Quote ③） ----------------------------

@dataclass
class StrokeGraphSample:
    """含真实人类笔画 + 逐笔空间语义输入 + 7 类成对空间关系边标签的图谱样本。"""

    stroke_seqs: np.ndarray     # [N_strokes, 24, SPATIAL_DIM]
    pair_indices: np.ndarray    # [N_pairs, 2]
    pair_geom: np.ndarray       # [N_pairs, 10]
    pair_labels: np.ndarray     # [N_pairs] in 0..6 (EDGE_RELATIONS)


def load_symbol_bank_from_dir(
    mw_root: str | Path,
    max_per_symbol: int = 80,
) -> Dict[str, List[List[List[Tuple[float, float]]]]]:
    """从 MathWriting `symbols/` 加载真实人类多笔画符号库，供 2D 结构合成提供真实书写变体。"""
    root_path = Path(mw_root)
    if (root_path / "mathwriting-2024").is_dir():
        root_path = root_path / "mathwriting-2024"
    sym_dir = root_path / "symbols"
    bank: Dict[str, List[List[List[Tuple[float, float]]]]] = {}
    if not sym_dir.is_dir():
        return bank

    target_syms = {"x", "y", "0", "1", "2", "A", "P", "e", "+", "-", "=", "|", "(", ")", "\\int", "\\sum", "\\lambda", "\\Phi"}
    for p in sorted(sym_dir.glob("*.inkml")):
        parsed = parse_inkml_bytes(p.read_bytes())
        if parsed is None:
            continue
        strokes, label = parsed
        key = label.strip()
        if key not in target_syms:
            continue
        bucket = bank.setdefault(key, [])
        if len(bucket) < max_per_symbol:
            bucket.append(strokes)
        if len(bank) >= len(target_syms) and all(len(v) >= max_per_symbol for v in bank.values()):
            break
    return bank


def _place_symbol_strokes(
    sym: str,
    ox: float,
    oy: float,
    h: float,
    rng: np.random.Generator,
    symbol_bank: Dict[str, List[List[List[Tuple[float, float]]]]],
) -> List[List[Tuple[float, float]]]:
    choices = symbol_bank.get(sym)
    if choices:
        raw = choices[int(rng.integers(0, len(choices)))]
        xs = [p[0] for s in raw for p in s]
        ys = [p[1] for s in raw for p in s]
        x0, y0 = min(xs), min(ys)
        bh = max(8.0, max(ys) - y0)
        sc = h / bh
        return [[(ox + (p[0] - x0) * sc, oy + (p[1] - y0) * sc) for p in s] for s in raw]
    return _prototype_glyph(sym, ox, oy, h, rng)


def generate_stroke_graph_samples(
    n_samples: int = 1200,
    seed: int = 2026,
    symbol_bank: Optional[Dict[str, List[List[List[Tuple[float, float]]]]]] = None,
    stroke_seq_len: int = 24,
    spatial_dim: int = SPATIAL_DIM,
) -> List[StrokeGraphSample]:
    """
    合成带真值逐对关系标签的 2D 大学数学表达样本（含 SAME_SYMBOL / RIGHT / ABOVE / BELOW / SUP / SUB / NONE）：
    - 分式 a / b (ABOVE / BELOW)
    - 积分上下限 \\int_{0}^{1} x (BELOW / ABOVE / RIGHT)
    - 上下标 x^{2} + x_{1} (SUP / SUB / RIGHT)
    - 特征方程 |A - \\lambda| = 0 (SAME_SYMBOL / RIGHT)
    """
    rng = np.random.default_rng(seed)
    bank = symbol_bank or {}
    outputs: List[StrokeGraphSample] = []

    for i in range(n_samples):
        h = float(rng.uniform(32.0, 58.0))
        ox = 40.0
        oy = 80.0
        # nodes: list of (strokes_list, sym_id_tag)
        placed_strokes: List[List[Tuple[float, float]]] = []
        sym_ids: List[int] = []
        rel_map: Dict[Tuple[int, int], int] = {}  # (sym_a, sym_b) -> rel_id

        def add_sym(sym_tok: str, sx: float, sy: float, sh: float) -> int:
            s_idx = len(sym_ids)
            sym_ids.append(s_idx)
            s_list = _place_symbol_strokes(sym_tok, sx, sy, sh, rng, bank)
            for st in s_list:
                placed_strokes.append(st)
            # map stroke -> symbol
            for _ in s_list:
                stroke_to_sym.append(s_idx)
            return s_idx

        stroke_to_sym: List[int] = []
        mode = i % 4

        if mode == 0:
            # 分式: num / bar / den + '=' + '0'
            bar_w = h * 1.45
            num_s = add_sym("x" if i % 2 == 0 else "1", ox + bar_w * 0.25, oy - h * 0.72, h * 0.75)
            bar_s = len(sym_ids)
            sym_ids.append(bar_s)
            placed_strokes.append([(ox, oy), (ox + bar_w, oy)])
            stroke_to_sym.append(bar_s)
            den_s = add_sym("2" if i % 2 == 0 else "x", ox + bar_w * 0.25, oy + h * 0.24, h * 0.75)
            eq_s = add_sym("=", ox + bar_w + h * 0.35, oy - h * 0.45, h * 0.9)
            rhs_s = add_sym("0", ox + bar_w + h * 1.3, oy - h * 0.45, h * 0.9)
            rel_map[(bar_s, num_s)] = EDGE_REL_TO_ID["ABOVE"]
            rel_map[(bar_s, den_s)] = EDGE_REL_TO_ID["BELOW"]
            rel_map[(bar_s, eq_s)] = EDGE_REL_TO_ID["RIGHT"]
            rel_map[(eq_s, rhs_s)] = EDGE_REL_TO_ID["RIGHT"]
        elif mode == 1:
            # 积分上下限: \int_{0}^{x} x
            int_s = add_sym("\\int", ox, oy - h * 0.15, h * 1.35)
            low_s = add_sym("0", ox + h * 0.55, oy + h * 0.72, h * 0.5)
            high_s = add_sym("1" if i % 2 == 0 else "x", ox + h * 0.62, oy - h * 0.35, h * 0.5)
            body_s = add_sym("x", ox + h * 1.25, oy, h * 0.95)
            rel_map[(int_s, low_s)] = EDGE_REL_TO_ID["BELOW"]
            rel_map[(int_s, high_s)] = EDGE_REL_TO_ID["ABOVE"]
            rel_map[(int_s, body_s)] = EDGE_REL_TO_ID["RIGHT"]
        elif mode == 2:
            # 上下标: x^{2} + A_{1}
            base1 = add_sym("x", ox, oy, h)
            sup1 = add_sym("2", ox + h * 0.8, oy - h * 0.42, h * 0.52)
            plus1 = add_sym("+", ox + h * 1.4, oy, h * 0.9)
            base2 = add_sym("A", ox + h * 2.25, oy, h)
            sub2 = add_sym("1", ox + h * 3.05, oy + h * 0.52, h * 0.52)
            rel_map[(base1, sup1)] = EDGE_REL_TO_ID["SUP"]
            rel_map[(base1, plus1)] = EDGE_REL_TO_ID["RIGHT"]
            rel_map[(plus1, base2)] = EDGE_REL_TO_ID["RIGHT"]
            rel_map[(base2, sub2)] = EDGE_REL_TO_ID["SUB"]
        else:
            # 矩阵方程: | A - \lambda | = 0
            toks = ["|", "A", "-", "\\lambda", "|", "=", "0"]
            cur_x = ox
            prev_s = -1
            for tok in toks:
                sid = add_sym(tok, cur_x, oy, h * 0.95)
                if prev_s >= 0:
                    rel_map[(prev_s, sid)] = EDGE_REL_TO_ID["RIGHT"]
                prev_s = sid
                cur_x += h * rng.uniform(0.75, 0.95)

        if len(placed_strokes) < 2:
            continue

        all_x = [p[0] for s in placed_strokes for p in s]
        all_y = [p[1] for s in placed_strokes for p in s]
        expr_scale = max(12.0, max(all_y) - min(all_y), 0.25 * (max(all_x) - min(all_x)))

        # Ensure shape [N_strokes, stroke_seq_len, spatial_dim] by padding/clipping
        padded_seqs = np.zeros((len(placed_strokes), stroke_seq_len, spatial_dim), dtype=np.float32)
        for idx_st, st in enumerate(placed_strokes):
            sp = arc_length_resample_strokes([st], max_len=stroke_seq_len, spatial_dim=spatial_dim)
            cn = min(len(sp), stroke_seq_len)
            padded_seqs[idx_st, :cn] = sp[:cn]

        pairs: List[Tuple[int, int]] = []
        geoms: List[np.ndarray] = []
        labels: List[int] = []
        n_st = len(placed_strokes)
        for sa in range(n_st):
            for sb in range(n_st):
                if sa == sb:
                    continue
                if abs(sa - sb) > 5 and stroke_to_sym[sa] != stroke_to_sym[sb]:
                    continue
                sym_a = stroke_to_sym[sa]
                sym_b = stroke_to_sym[sb]
                if sym_a == sym_b:
                    rel_id = EDGE_REL_TO_ID["SAME_SYMBOL"]
                else:
                    rel_id = rel_map.get((sym_a, sym_b), EDGE_REL_TO_ID["NONE"])
                pairs.append((sa, sb))
                geoms.append(extract_pair_geom_10(placed_strokes[sa], placed_strokes[sb], expr_scale))
                labels.append(rel_id)

        if pairs:
            outputs.append(
                StrokeGraphSample(
                    stroke_seqs=padded_seqs,
                    pair_indices=np.asarray(pairs, dtype=np.int64),
                    pair_geom=np.stack(geoms, axis=0),
                    pair_labels=np.asarray(labels, dtype=np.int64),
                )
            )

    return outputs
