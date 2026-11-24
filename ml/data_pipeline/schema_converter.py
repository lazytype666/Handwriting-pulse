"""
《笔迹脉搏》双流数据管线与 6 类认知状态转换器（14 维因果动力学与解题图谱对照融合版）。

【核心架构特性】
1. 彻底解耦 60~120Hz 触控采样率：按物理弧长/真实时间戳还原无采样率依赖速度比、停顿 Z 分数、多时间常数指数衰减积分；
2. DYNAMICS_DIM = 14：
   - 0..9：笔画时长、个人停顿Z、速度比、速度波动、前序动量、橡皮、微笔、短停能量、涂改能量、方向对齐 A(t)
   - 10..13（图四新增直连图谱状态）：[anchorGap, stepStagnation, stepAdvancePulse, stepProgress]
3. 空间端：每笔生成 10 通道等弧长序列 [L_s, 10] -> EdgeMathGraphNet 提取语义 Z_sem[j]；
4. 6 类认知状态：
   0=FLUENT_DERIVATION, 1=DEEP_THINKING_HOLD, 2=CALC_MICRO_HESITATION,
   3=STUCK_CALCULATION, 4=STUCK_CONCEPT_GAP, 5=DIRECTION_DRIFT_PITFALL
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mathwriting_loader import SPATIAL_DIM, arc_length_resample_strokes

CLASS_NAMES = [
    "FLUENT_DERIVATION",
    "DEEP_THINKING_HOLD",
    "CALC_MICRO_HESITATION",
    "STUCK_CALCULATION",
    "STUCK_CONCEPT_GAP",
    "DIRECTION_DRIFT_PITFALL",
]

LABEL_TO_CLASS: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DYNAMICS_DIM = 14
STROKE_MAX_PTS = 24
WINDOW_STEPS = 20


@dataclass
class StrokeWindowSample:
    """单次卡壳决策窗口的双流对齐张量与监督标签。"""

    stroke_spatial_seqs: np.ndarray  # [W, L_s, 10] 每一决策步的等弧长几何笔迹
    spatial_seq: np.ndarray          # [L_expr, 10] 整体窗口表达式几何流
    dynamics_seq: np.ndarray         # [W, 14] 笔画步级动力学 + 图谱状态流
    label_id: int
    label_name: str
    writer_profile: str
    raw_pause_ms: float
    personal_pause_z: float
    erase_density: float
    boundary_flag: float
    speed_ratio: float
    latent_m: float
    latent_r: float
    latent_a: float


def load_inkpulse_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_stroke_segment_dynamics(
    strokes: Sequence[Dict[str, Any]],
    baseline_pause_log_mean: float = math.log(1200.0),
    baseline_pause_log_std: float = 0.70,
    baseline_speed_px_sec: float = 160.0,
    alignment_trace: Optional[Sequence[float]] = None,
    graph_trace: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将原始笔画列表转为“决策步级对齐”的双流特征：
    返回:
      per_stroke_spatial: [N_strokes, STROKE_MAX_PTS, 10]
      dynamics_steps:     [N_strokes, 14]
      stroke_end_times:   [N_strokes]
      stroke_raw_meta:    [N_strokes, 3] -> (raw_pause_ms, boundary_flag, erase_flag)
    """
    n_strokes = len(strokes)
    if n_strokes == 0:
        return (
            np.zeros((1, STROKE_MAX_PTS, SPATIAL_DIM), dtype=np.float32),
            np.zeros((1, DYNAMICS_DIM), dtype=np.float32),
            np.zeros((1,), dtype=np.int64),
            np.zeros((1, 3), dtype=np.float32),
        )

    per_stroke_spatial = np.zeros((n_strokes, STROKE_MAX_PTS, SPATIAL_DIM), dtype=np.float32)
    dynamics_steps = np.zeros((n_strokes, DYNAMICS_DIM), dtype=np.float32)
    stroke_end_times = np.zeros((n_strokes,), dtype=np.int64)
    stroke_raw_meta = np.zeros((n_strokes, 3), dtype=np.float32)

    prev_end_t: Optional[int] = None
    running_momentum = 0.65
    decay_pause = 0.0
    decay_erase = 0.0

    for j, s in enumerate(strokes):
        pts = s.get("points", [])
        if not pts:
            continue

        is_eraser = 1.0 if s.get("tool") == "eraser" else 0.0
        t_start = int(pts[0]["t"])
        t_end = int(pts[-1]["t"])
        dur_ms = max(16.0, float(t_end - t_start))
        pause_before = max(0.0, float(t_start - prev_end_t)) if prev_end_t is not None else 0.0
        dt_since_prev = pause_before + dur_ms

        arc_len = 0.0
        seg_speeds: List[float] = []
        xs = [float(pts[0]["x"])]
        ys = [float(pts[0]["y"])]
        for k in range(1, len(pts)):
            xk, yk, tk = float(pts[k]["x"]), float(pts[k]["y"]), int(pts[k]["t"])
            xp, yp, tp = float(pts[k - 1]["x"]), float(pts[k - 1]["y"]), int(pts[k - 1]["t"])
            xs.append(xk)
            ys.append(yk)
            d = math.hypot(xk - xp, yk - yp)
            arc_len += d
            dt_k = max(4.0, float(tk - tp)) / 1000.0
            seg_speeds.append(min(2000.0, d / dt_k))

        mean_speed = (arc_len / (dur_ms / 1000.0)) if dur_ms > 0 else 0.0
        speed_ratio = min(2.5, mean_speed / max(20.0, baseline_speed_px_sec))
        speed_instab = (
            min(2.0, float(np.std(seg_speeds)) / max(20.0, float(np.mean(seg_speeds))))
            if len(seg_speeds) >= 2
            else 0.0
        )

        pause_ref = max(pause_before, 120.0)
        pause_z = (
            float(max(-2.5, min(4.0, (math.log(pause_ref) - baseline_pause_log_mean) / max(0.35, baseline_pause_log_std))))
            if pause_before > 180.0
            else -1.0
        )

        bw = max(1.0, max(xs) - min(xs))
        bh = max(1.0, max(ys) - min(ys))
        is_horiz = bw > bh * 3.2 and bw >= 16.0
        is_tall = bh >= 36.0 and bh > bw * 2.2
        boundary_flag = 1.0 if (not is_eraser and (is_horiz or is_tall)) else 0.0

        is_frag = 1.0 if (not is_eraser and arc_len < 18.0 and dur_ms < 220.0) else 0.0

        decay_pause = decay_pause * math.exp(-dt_since_prev / 1500.0) + max(0.0, pause_z)
        decay_erase = decay_erase * math.exp(-dt_since_prev / 10000.0) + (is_eraser + 0.35 * is_frag)

        pre_pause_m = running_momentum
        running_momentum = float(
            max(0.05, min(1.0, 0.55 * running_momentum + 0.45 * min(1.0, speed_ratio) * math.exp(-pause_before / 4500.0)))
        )

        align_a = float(alignment_trace[j]) if alignment_trace and j < len(alignment_trace) else 0.0
        if graph_trace and j < len(graph_trace):
            a_gap, st_stag, adv_pulse, st_prog = graph_trace[j]
        else:
            a_gap, st_stag, adv_pulse, st_prog = (0.35, 0.0, 0.0, min(1.0, (j + 1) / max(1.0, float(n_strokes))))

        dynamics_steps[j] = [
            min(1.5, math.log1p(dur_ms) / 8.0),
            pause_z,
            speed_ratio,
            speed_instab,
            pre_pause_m,
            is_eraser,
            is_frag,
            math.tanh(decay_pause / 2.5),
            math.tanh(decay_erase / 2.5),
            max(-1.0, min(1.0, align_a)),
            max(0.0, min(1.0, a_gap)),
            max(0.0, min(1.0, st_stag)),
            max(0.0, min(1.0, adv_pulse)),
            max(0.0, min(1.0, st_prog)),
        ]

        xy = [(float(p["x"]), float(p["y"])) for p in pts]
        sp = arc_length_resample_strokes([xy], max_len=STROKE_MAX_PTS, spatial_dim=SPATIAL_DIM)
        copy_n = min(len(sp), STROKE_MAX_PTS)
        per_stroke_spatial[j, :copy_n] = sp[:copy_n]

        stroke_end_times[j] = t_end
        stroke_raw_meta[j] = [pause_before, boundary_flag, is_eraser]
        prev_end_t = t_end

    return per_stroke_spatial, dynamics_steps, stroke_end_times, stroke_raw_meta


def _pad_step_window(arr: np.ndarray, win_steps: int) -> np.ndarray:
    if len(arr) >= win_steps:
        return arr[-win_steps:]
    pad_shape = (win_steps - len(arr),) + arr.shape[1:]
    return np.vstack([np.zeros(pad_shape, dtype=arr.dtype), arr])


def convert_session_to_windows(
    session: Dict[str, Any],
    window_steps: int = WINDOW_STEPS,
    writer_profile: str = "real_user",
) -> List[StrokeWindowSample]:
    """将导出的 v2.0-annotated 会话切分为 6 类认知状态训练/验证窗口（含 14 维图谱状态特征）。"""
    strokes = session.get("strokes", [])
    annotations = session.get("annotations", [])
    cognitive_meta = session.get("cognitiveSession", {})
    if not annotations and isinstance(cognitive_meta, dict):
        annotations = cognitive_meta.get("annotations", [])

    fp = cognitive_meta.get("fingerprint") or cognitive_meta.get("cognitiveState") or {}
    log_mean = float(fp.get("pauseLogMean", math.log(1200.0)))
    log_std = float(fp.get("pauseLogStd", 0.70))
    base_speed = float(fp.get("baselineSpeedPxPerSec", 160.0))

    state_trace = session.get("stateTrace") or cognitive_meta.get("stateTrace") or []
    align_per_stroke: List[float] = []
    graph_per_stroke: List[Tuple[float, float, float, float]] = []
    if state_trace:
        for s in strokes:
            pts = s.get("points", [])
            t_end = int(pts[-1]["t"]) if pts else 0
            closest = min(state_trace, key=lambda item: abs(int(item.get("t", 0)) - t_end))
            align_per_stroke.append(float(closest.get("A", 0.0)))
            graph_per_stroke.append((
                float(closest.get("anchorGap", 0.35)),
                float(closest.get("stepStagnation", 0.0)),
                float(closest.get("stepAdvancePulse", 0.0)),
                float(closest.get("stepProgress", 0.2)),
            ))

    per_stroke_sp, dyn_steps, stroke_ts, raw_meta = extract_stroke_segment_dynamics(
        strokes,
        baseline_pause_log_mean=log_mean,
        baseline_pause_log_std=log_std,
        baseline_speed_px_sec=base_speed,
        alignment_trace=align_per_stroke if align_per_stroke else None,
        graph_trace=graph_per_stroke if graph_per_stroke else None,
    )
    if len(stroke_ts) < 2:
        return []

    samples: List[StrokeWindowSample] = []
    for ann in annotations:
        if "INVALID_INTERRUPT" in str(ann.get("note", "")):
            continue
        raw_label = str(ann.get("label", ""))
        if raw_label == "SELF_CORRECTION":
            mapped_label = "CALC_MICRO_HESITATION"
        elif raw_label == "HINT_EFFECTIVE":
            continue
        elif raw_label == "PITFALL_TRIGGERED":
            mapped_label = "DIRECTION_DRIFT_PITFALL"
        elif raw_label in LABEL_TO_CLASS:
            mapped_label = raw_label
        else:
            continue

        target_t = int(ann.get("timestamp", stroke_ts[-1]))
        end_step = int(np.searchsorted(stroke_ts, target_t, side="right"))
        start_step = max(0, end_step - window_steps)
        if end_step - start_step < 2:
            continue

        win_sp_steps = _pad_step_window(per_stroke_sp[start_step:end_step], window_steps)
        win_dyn = _pad_step_window(dyn_steps[start_step:end_step], window_steps)

        win_xy = [
            [(float(p["x"]), float(p["y"])) for p in s.get("points", [])]
            for s in strokes[start_step:end_step]
            if s.get("tool") != "eraser" and s.get("points")
        ]
        expr_spatial = arc_length_resample_strokes(win_xy, max_len=64, spatial_dim=SPATIAL_DIM)

        tail_meta = raw_meta[max(start_step, end_step - 4) : end_step]
        tail_dyn = win_dyn[-min(4, window_steps) :]

        if mapped_label == "FLUENT_DERIVATION":
            raw_pause_ms = float(np.max(tail_meta[:, 0]))
        else:
            raw_pause_ms = float(ann.get("durationMs") or float(np.max(tail_meta[:, 0])))

        bnd_window = raw_meta[max(start_step, end_step - 3) : end_step, 1]
        boundary_flag = float(np.max(bnd_window)) if len(bnd_window) > 0 else 0.0
        pause_z = float(np.max(tail_dyn[:, 1]))
        erase_density = float(np.mean(win_dyn[-min(8, window_steps) :, 5]))
        speed_ratio = float(np.mean(tail_dyn[:, 2]))

        m_val, r_val, a_val = 0.5, 0.2, 0.0
        if state_trace:
            closest = min(state_trace, key=lambda item: abs(int(item.get("t", 0)) - target_t))
            m_val = float(closest.get("M", 0.5))
            r_val = float(closest.get("R", 0.2))
            a_val = float(closest.get("A", 0.0))

        samples.append(
            StrokeWindowSample(
                stroke_spatial_seqs=win_sp_steps,
                spatial_seq=expr_spatial,
                dynamics_seq=win_dyn,
                label_id=LABEL_TO_CLASS[mapped_label],
                label_name=mapped_label,
                writer_profile=writer_profile,
                raw_pause_ms=raw_pause_ms,
                personal_pause_z=pause_z,
                erase_density=erase_density,
                boundary_flag=boundary_flag,
                speed_ratio=speed_ratio,
                latent_m=m_val,
                latent_r=r_val,
                latent_a=a_val,
            )
        )

    return samples


def generate_synthetic_annotated_sessions(
    n_sessions: int = 80,
    seed: int = 2026,
) -> List[Tuple[Dict[str, Any], str]]:
    """
    生成覆盖 6 类大学数学做题状态的快慢手对比会话集（同步附带 4 维图谱对照特征 stateTrace）。
    """
    rng = np.random.default_rng(seed)
    outputs: List[Tuple[Dict[str, Any], str]] = []
    base_t = 1767225600000

    for sess_idx in range(n_sessions):
        is_slow = sess_idx % 2 == 1
        writer_profile = "slow_writer" if is_slow else "fast_writer"
        personal_pause_ms = float(rng.uniform(2050.0, 2550.0)) if is_slow else float(rng.uniform(680.0, 920.0))
        log_mean = math.log(personal_pause_ms)
        log_std = 0.36
        base_speed = float(rng.uniform(95.0, 125.0)) if is_slow else float(rng.uniform(205.0, 255.0))

        t_cursor = base_t + sess_idx * 180_000
        strokes: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        state_trace: List[Dict[str, Any]] = []
        stroke_id = 1

        def add_stroke(
            tool: str,
            start_x: float,
            y: float,
            dx_step: float,
            n_pts: int,
            dt_step: int,
            shape: str = "digit",
            jitter_speed: float = 0.0,
            graph_vals: Tuple[float, float, float, float] = (0.2, 0.0, 0.0, 0.3),
        ) -> None:
            nonlocal t_cursor, stroke_id
            pts = []
            x_cur = start_x
            for k in range(n_pts):
                step_dt = max(8, int(dt_step * (1.0 + float(rng.normal(0, jitter_speed)))))
                if shape == "eq":
                    vy = 0.0
                    dx_k = dx_step
                elif shape == "int":
                    vy = (k - n_pts * 0.5) * 8.5
                    dx_k = -1.2 if (k < 2 or k >= n_pts - 2) else 0.0
                else:
                    vy = math.sin(k * 0.95) * 11.0 + float(rng.normal(0, 1.0))
                    dx_k = dx_step
                x_cur += dx_k
                pts.append({"x": round(x_cur, 2), "y": round(y + vy, 2), "t": int(t_cursor)})
                t_cursor += step_dt
            obj: Dict[str, Any] = {"id": stroke_id, "points": pts}
            if tool == "eraser":
                obj["tool"] = "eraser"
            strokes.append(obj)
            stroke_id += 1
            ag, ss, ap, sp_prog = graph_vals
            state_trace.append({
                "t": t_cursor,
                "M": 0.65,
                "R": 0.25,
                "A": 0.35 if ss < 0.4 else -0.1,
                "anchorGap": ag,
                "stepStagnation": ss,
                "stepAdvancePulse": ap,
                "stepProgress": sp_prog,
            })

        # 类 0：FLUENT_DERIVATION（低 anchorGap、步进脉冲正常跃迁）
        for b in range(5):
            t_cursor += int(personal_pause_ms * float(rng.uniform(0.55, 1.15)))
            add_stroke("pen", 50 + b * 24, 100, 3.8 if not is_slow else 2.2, 8, 16, graph_vals=(0.15, 0.05, 0.65 if b == 4 else 0.1, 0.33))
        if float(rng.random()) < 0.12:
            t_cursor += 160
            add_stroke("eraser", 120, 100, 1.5, 4, 16, graph_vals=(0.18, 0.10, 0.0, 0.33))
            t_cursor += 140
            add_stroke("pen", 120, 100, 3.4 if not is_slow else 2.1, 6, 16, graph_vals=(0.12, 0.05, 0.45, 0.35))
        annotations.append({"id": "a0", "timestamp": t_cursor, "label": "FLUENT_DERIVATION", "source": "auto_fluent", "durationMs": 4800})
        state_trace[-1].update({"M": 0.88, "R": 0.08, "A": 0.55})

        # 类 1：DEEP_THINKING_HOLD（步骤跃迁刚完成后自然审题停顿：stepAdvancePulse 高 + 无涂改）
        for b in range(3):
            t_cursor += int(personal_pause_ms * float(rng.uniform(0.65, 0.95)))
            ambig = (b == 2 and float(rng.random()) < 0.20)
            add_stroke("pen", 60 + b * 24, 160, 3.6 if not is_slow else 2.2, 8, int(rng.integers(15, 20)), shape="eq" if ambig else "digit", graph_vals=(0.18, 0.08, 0.85 if b == 2 else 0.2, 0.50))
        think_pause = int(personal_pause_ms * float(rng.uniform(1.4, 2.35)))
        t_cursor += think_pause
        add_stroke("pen", 145, 160, 3.2 if not is_slow else 2.0, 6, 16, graph_vals=(0.22, 0.12, 0.65, 0.52))
        annotations.append({"id": "a1", "timestamp": t_cursor, "label": "DEEP_THINKING_HOLD", "source": "user_tap_button", "durationMs": think_pause})
        state_trace[-1].update({"M": 0.48, "R": 0.25, "A": 0.30})

        # 类 2：CALC_MICRO_HESITATION（微卡犹豫：中度 anchorGap + 轻微擦改）
        t_cursor += int(personal_pause_ms * float(rng.uniform(0.85, 1.15)))
        add_stroke("pen", 80, 220, 2.0, 6, 22, jitter_speed=0.35, graph_vals=(0.38, 0.32, 0.05, 0.55))
        t_cursor += int(personal_pause_ms * float(rng.uniform(0.9, 1.45)))
        add_stroke("eraser", 80, 220, 1.8, 5, 18, graph_vals=(0.40, 0.42, 0.0, 0.55))
        t_cursor += 280
        add_stroke("pen", 82, 220, 2.4, 6, 18, jitter_speed=0.25, graph_vals=(0.35, 0.36, 0.05, 0.55))
        annotations.append({"id": "a2", "timestamp": t_cursor, "label": "CALC_MICRO_HESITATION", "source": "auto_fluent", "durationMs": int(personal_pause_ms * 1.15)})
        state_trace[-1].update({"M": 0.42, "R": 0.48, "A": 0.10})

        # 类 3：STUCK_CALCULATION（核心符号已在画布 anchorGap<=0.32，但同一步停滞擦写 stepStagnation>=0.75）
        t_cursor += int(personal_pause_ms * 1.1)
        add_stroke("pen", 90, 280, 0.9, 6, 28, jitter_speed=0.45, graph_vals=(0.26, 0.62, 0.0, 0.62))
        t_cursor += int(personal_pause_ms * float(rng.uniform(1.4, 2.2)))
        add_stroke("eraser", 90, 280, 1.4, 6, 20, graph_vals=(0.28, 0.78, 0.0, 0.62))
        t_cursor += 350
        add_stroke("pen", 92, 280, 0.8, 4, 30, jitter_speed=0.45, graph_vals=(0.25, 0.82, 0.0, 0.62))
        t_cursor += 320
        add_stroke("eraser", 92, 280, 1.5, 6, 20, graph_vals=(0.28, 0.88, 0.0, 0.62))
        t_cursor += int(rng.integers(300, 600))
        add_stroke("pen", 95, 280, 0.85, 5, 29, jitter_speed=0.4, graph_vals=(0.24, 0.86, 0.0, 0.62))
        annotations.append({"id": "a3", "timestamp": t_cursor, "label": "STUCK_CALCULATION", "source": "user_tap_button", "durationMs": int(personal_pause_ms * 1.9)})
        state_trace[-1].update({"M": 0.16, "R": 0.88, "A": -0.15})

        # 类 4：STUCK_CONCEPT_GAP（思路断链：anchorGap>=0.65 + 长停顿，缺少当前步骤关键算符）
        t_cursor += 500
        bnd_shape = "int" if float(rng.random()) < 0.5 else ("eq" if float(rng.random()) < 0.75 else "digit")
        add_stroke("pen", 110, 350, 1.9 if not is_slow else 1.2, 8, int(rng.integers(26, 36)), shape=bnd_shape, jitter_speed=0.25, graph_vals=(0.72, 0.68, 0.0, 0.25))
        concept_pause = int(personal_pause_ms * float(rng.uniform(2.05, 3.4)))
        t_cursor += concept_pause
        add_stroke("pen", 145, 350, 1.5, 4, 24, graph_vals=(0.76, 0.82, 0.0, 0.25))
        annotations.append({"id": "a4", "timestamp": t_cursor, "label": "STUCK_CONCEPT_GAP", "source": "user_tap_button", "durationMs": concept_pause})
        state_trace[-1].update({"M": 0.14, "R": 0.86, "A": -0.10})

        # 类 5：DIRECTION_DRIFT_PITFALL（歧路偏离：连续写出偏离符号且 A=-0.65）
        t_cursor += 500
        add_stroke("pen", 100, 420, 3.2, 7, 17, shape="eq", graph_vals=(0.55, 0.45, 0.0, 0.45))
        t_cursor += int(personal_pause_ms * 0.9)
        add_stroke("pen", 130, 420, 2.8, 7, 18, shape="digit", graph_vals=(0.62, 0.52, 0.0, 0.45))
        annotations.append({"id": "a5", "timestamp": t_cursor, "label": "DIRECTION_DRIFT_PITFALL", "source": "syntax_pitfall", "durationMs": 950})
        state_trace[-1].update({"M": 0.58, "R": 0.62, "A": -0.65})

        outputs.append((
            {
                "meta": {"version": "2.0-annotated", "exportedAt": "2026-11-24T00:00:00.000Z", "problemId": f"syn6_{sess_idx:03d}", "strokeCount": len(strokes)},
                "strokes": strokes,
                "stallEvents": [],
                "annotations": annotations,
                "stateTrace": state_trace,
                "cognitiveSession": {
                    "fingerprint": {"pauseLogMean": log_mean, "pauseLogStd": log_std, "baselineSpeedPxPerSec": base_speed},
                    "annotations": annotations,
                    "stateTrace": state_trace,
                },
            },
            writer_profile,
        ))
    return outputs
