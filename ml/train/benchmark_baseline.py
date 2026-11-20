"""
《笔迹脉搏》纯 NumPy / Scikit-Learn 6 类认知状态消融基准评测脚本。

对比验证：
1. 阶段二固定阈值（STALL=3000ms）；
2. 端侧多门控状态机 [M, R, A] + 个人分位数 + 心算保护阻尼（零训练纯端侧基线）；
3. 决策步级解耦双流聚合特征（5 折分层交叉验证 6 分类识别 + 深度心算防误报率）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data_pipeline.schema_converter import (  # noqa: E402
    CLASS_NAMES,
    StrokeWindowSample,
    convert_session_to_windows,
    generate_synthetic_annotated_sessions,
    load_inkpulse_json,
)


def load_samples(data_dir: str | None) -> List[StrokeWindowSample]:
    samples: List[StrokeWindowSample] = []
    if data_dir and Path(data_dir).exists():
        json_files = sorted(Path(data_dir).glob("*.json"))
        for jf in json_files:
            samples.extend(convert_session_to_windows(load_inkpulse_json(jf), writer_profile="real_user"))
        print(f"[数据加载] 从 {data_dir} 载入 {len(json_files)} 个导出会话，共 {len(samples)} 个窗口。")

    if not samples:
        sessions = generate_synthetic_annotated_sessions(n_sessions=80, seed=2026)
        for sess_obj, profile in sessions:
            samples.extend(convert_session_to_windows(sess_obj, writer_profile=profile))
        print(f"[数据生成] 使用 80 轮快慢手 6 分类对照集，共 {len(samples)} 个决策步时间窗。")
    return samples


def evaluate_binary_stuck(y_true_6c: np.ndarray, y_pred_binary: np.ndarray) -> Dict[str, float]:
    """
    真实需要干预卡壳：label >= 2 (微卡 / 算式受阻 / 思路卡壳 / 歧路偏离)。
    重点衡量深度心算 (label == 1) 的误报率 (FPR)。
    """
    y_true_stuck = (y_true_6c >= 2).astype(int)
    f1 = float(f1_score(y_true_stuck, y_pred_binary, zero_division=0))
    acc = float(accuracy_score(y_true_stuck, y_pred_binary))
    think_mask = y_true_6c == 1
    think_fpr = float(np.mean(y_pred_binary[think_mask])) if np.any(think_mask) else 0.0
    return {"acc": acc, "stuck_f1": f1, "think_fpr": think_fpr}


def extract_tabular_dual_stream_vector(sample: StrokeWindowSample) -> np.ndarray:
    """从 6 通道重采样空间流 + 10 维决策步级动力学流提取纯因果统计特征（零未来泄漏）。"""
    sp = sample.spatial_seq
    dy = sample.dynamics_seq
    last_step = dy[-1]
    prev_slice = dy[-min(5, len(dy)) : -1] if len(dy) >= 2 else dy
    return np.asarray(
        [
            float(last_step[0]),                          # 最近一笔对数时长
            float(np.max(dy[-4:, 1])),                    # 近 4 笔最大个人 Z 停顿
            float(last_step[1]),                          # 当前笔前个人 Z 停顿
            float(last_step[2]),                          # 当前速度比
            float(np.mean(prev_slice[:, 2])),             # 前序平均速度比
            float(np.max(dy[-4:, 3])),                    # 近期笔内速度震荡波动率
            float(last_step[4]),                          # 停顿前惯性动量
            float(np.mean(dy[-6:, 5])),                   # 近期橡皮擦密度
            float(np.max(dy[-4:, 6])),                    # 短碎微笔比例
            float(last_step[7]),                          # 短时停顿衰变能量 τ=1.5s
            float(last_step[8]),                          # 中期涂改衰变能量 τ=10s
            float(last_step[9]),                          # 语法语义对齐度 A(t)
            sample.boundary_flag,                         # 是否在逻辑/微积分/矩阵边界附近
            float(np.mean(np.hypot(sp[:, 0], sp[:, 1]))), # 几何步长均值
            float(np.std(sp[:, 2])),                      # 水平角度方差
            float(np.std(sp[:, 3])),                      # 竖直角度方差
        ],
        dtype=np.float32,
    )


def run_benchmark(data_dir: str | None = None) -> None:
    samples = load_samples(data_dir)
    y_true = np.asarray([s.label_id for s in samples], dtype=int)

    # 1. 基线 A：固定阈值 3000ms
    pred_fixed = np.asarray(
        [1 if (s.raw_pause_ms >= 3000.0 or (s.erase_density >= 0.08 and s.raw_pause_ms >= 1500.0)) else 0 for s in samples],
        dtype=int,
    )
    res_fixed = evaluate_binary_stuck(y_true, pred_fixed)

    # 2. 基线 B：多门控端侧状态机
    def score_multi_gate(s: StrokeWindowSample) -> int:
        dy = s.dynamics_seq
        pre_speed = float(np.mean(dy[-min(5, len(dy)) : -1, 2])) if len(dy) >= 2 else s.speed_ratio
        speed_instab = float(np.max(dy[-min(4, len(dy)) :, 3]))
        decay_e = float(dy[-1, 8])
        align_a = float(dy[-1, 9])

        if align_a <= -0.45:
            return 1
        if decay_e >= 0.35 or (s.erase_density >= 0.1 and (speed_instab >= 0.18 or s.personal_pause_z >= 0.5)):
            return 1
        if s.erase_density < 0.05 and s.raw_pause_ms < 1400.0:
            return 0
        boundary_mult = 1.42 if s.boundary_flag > 0.5 else 0.78
        in_shield = s.boundary_flag < 0.5 and pre_speed >= 0.9 and s.personal_pause_z < 2.35
        shield = 0.65 if in_shield else 1.0
        hesitation_boost = max(0.0, 0.9 - pre_speed) * 0.9
        if s.boundary_flag > 0.5 and pre_speed < 0.95 and s.personal_pause_z >= 1.55:
            return 1
        if max(0.0, s.personal_pause_z) * boundary_mult * shield + hesitation_boost >= 1.95:
            return 1
        return 0

    pred_adaptive = np.asarray([score_multi_gate(s) for s in samples], dtype=int)
    res_adaptive = evaluate_binary_stuck(y_true, pred_adaptive)

    # 3. 决策步级解耦双流融合（6 分类 5 折交叉验证）
    x_mat = np.vstack([extract_tabular_dual_stream_vector(s) for s in samples])
    oof_pred = np.zeros_like(y_true)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(x_mat, y_true):
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=600, random_state=42),
        )
        clf.fit(x_mat[tr_idx], y_true[tr_idx])
        oof_pred[va_idx] = clf.predict(x_mat[va_idx])

    res_dual = evaluate_binary_stuck(y_true, (oof_pred >= 2).astype(int))
    macro_f1_6c = float(f1_score(y_true, oof_pred, average="macro", zero_division=0))
    acc_6c = float(accuracy_score(y_true, oof_pred))

    print("\n================ 《笔迹脉搏》6 分类确认架构消融基准结果 ================")
    print(f"{'模型策略':<34} | {'需干预 F1':<11} | {'二分类准确率':<11} | {'深度心算误报率(越低越好)'}")
    print("-" * 82)
    print(f"{'1. 阶段二固定阈值 (STALL=3000ms)':<34} | {res_fixed['stuck_f1']:.4f}      | {res_fixed['acc']:.4f}      | {res_fixed['think_fpr']*100:6.2f}%")
    print(f"{'2. 多门控端侧状态机 [M,R,A]':<34} | {res_adaptive['stuck_f1']:.4f}      | {res_adaptive['acc']:.4f}      | {res_adaptive['think_fpr']*100:6.2f}%")
    print(f"{'3. 步级解耦双流融合 (5折CV六分类)':<34} | {res_dual['stuck_f1']:.4f}      | {res_dual['acc']:.4f}      | {res_dual['think_fpr']*100:6.2f}%")
    print("-" * 82)
    print(f"6 分类细粒度鉴别 ({', '.join(CLASS_NAMES)}) -> 准确率: {acc_6c:.4f}, Macro-F1: {macro_f1_6c:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="《笔迹脉搏》6 类认知状态消融基准评估脚本")
    parser.add_argument("--data-dir", type=str, default=None, help="App 导出的 JSON 目录（可选）")
    args = parser.parse_args()
    run_benchmark(args.data_dir)
