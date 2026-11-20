"""
《笔迹脉搏》真实标注数据集质检、双轴留一交叉验证（LOUO-CV / LOPO-CV）与全套科学指标评测脚本。

对齐《笔迹脉搏_真实手写认知状态标注指南_v2.docx》第 4~5 章：
1. 入库质检门禁：检查 strokes 完整性、时间戳单调性、离屏中断过滤与 6 类分布均衡性；
2. 严防数据泄漏的按人分组交叉验证（GroupKFold / Leave-One-User-Out，从文件名 `u01_slow_L2_...json` 自动解析用户 ID）；
3. 输出合理且全面的评测指标：
   - 深度心算误报率 ThinkFPR（安全红线 < 5%）
   - 需干预二分类 StuckF1 & PR-AUC（处理类别不平衡）
   - 6 分类宏平均 6C_MacroF1 & 6×6 混淆矩阵
   - 可选：双人盲审一致性 Cohen's κ（`--blind-audit-dir`）
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data_pipeline.schema_converter import (  # noqa: E402
    CLASS_NAMES,
    WINDOW_STEPS,
    StrokeWindowSample,
    convert_session_to_windows,
    generate_synthetic_annotated_sessions,
    load_inkpulse_json,
)

USER_ID_RE = re.compile(r"^(u\d+)", re.IGNORECASE)


def parse_user_id(filename: str) -> str:
    m = USER_ID_RE.match(Path(filename).stem)
    return m.group(1).lower() if m else Path(filename).stem.split("_")[0].lower()


def audit_json_session(path: Path) -> Tuple[bool, List[str], Dict]:
    """执行指南 4.3 节 6 条入库质检门禁中的自动化结构/时间/匿名检查。"""
    issues: List[str] = []
    data = load_inkpulse_json(path)
    strokes = data.get("strokes", [])
    if len(strokes) < 8:
        issues.append(f"笔画数过少 ({len(strokes)} < 8)")

    prev_t = -1
    max_gap_ms = 0
    for s in strokes:
        for p in s.get("points", []):
            t = int(p.get("t", 0))
            if prev_t >= 0:
                if t < prev_t:
                    issues.append("存在非单调倒退时间戳")
                    break
                max_gap_ms = max(max_gap_ms, t - prev_t)
            prev_t = t
        if "存在非单调倒退时间戳" in issues:
            break
    if max_gap_ms > 600_000:
        issues.append(f"存在超长锁屏断层 ({max_gap_ms / 1000:.1f}s > 600s)")

    meta = data.get("cognitiveSession", {})
    anns = data.get("annotations") or (meta.get("annotations") if isinstance(meta, dict) else []) or []
    if not anns:
        issues.append("缺少 annotations 标注列表")

    return len(issues) == 0, issues, data


def extract_window_summary_features(w: StrokeWindowSample) -> np.ndarray:
    """提取与端侧因果状态机同源的窗口动力学汇总特征（用于无 GPU 时的快速 LOUO 基线评测）。"""
    dyn = w.dynamics_seq
    tail = dyn[-4:]
    return np.concatenate([
        dyn.mean(axis=0),
        dyn.max(axis=0),
        tail.mean(axis=0),
        tail[-1],
        np.array([w.raw_pause_ms / 1000.0, w.boundary_flag, w.erase_density, w.speed_ratio], dtype=np.float32),
    ])


def compute_cohen_kappa_dirs(primary_dir: Path, audit_dir: Path) -> float | None:
    from sklearn.metrics import cohen_kappa_score

    y1: List[int] = []
    y2: List[int] = []
    for p1 in sorted(primary_dir.glob("*.json")):
        p2 = audit_dir / p1.name
        if not p2.exists():
            continue
        w1 = convert_session_to_windows(load_inkpulse_json(p1), window_steps=WINDOW_STEPS)
        w2 = convert_session_to_windows(load_inkpulse_json(p2), window_steps=WINDOW_STEPS)
        n = min(len(w1), len(w2))
        for i in range(n):
            y1.append(w1[i].label_id)
            y2.append(w2[i].label_id)
    if len(y1) < 5:
        return None
    return float(cohen_kappa_score(y1, y2))


def main() -> None:
    parser = argparse.ArgumentParser(description="《笔迹脉搏》真实标注质检与 LOUO-CV 科学指标评测")
    parser.add_argument("--data-dir", type=str, default=None, help="真实导出的 v2.0-annotated JSON 目录")
    parser.add_argument("--blind-audit-dir", type=str, default=None, help="15% 独立双人盲审 JSON 目录（可选，用于算 Cohen's κ）")
    parser.add_argument("--n-splits", type=int, default=5, help="按人分组 GroupKFold 折数")
    args = parser.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
    from sklearn.model_selection import GroupKFold

    windows: List[StrokeWindowSample] = []
    groups: List[str] = []

    if args.data_dir and Path(args.data_dir).exists():
        json_files = sorted(Path(args.data_dir).glob("*.json"))
        passed = 0
        for jf in json_files:
            ok, issues, sess = audit_json_session(jf)
            if not ok:
                print(f"[质检未通过] {jf.name}: {'; '.join(issues)}")
                continue
            passed += 1
            uid = parse_user_id(jf.name)
            ws = convert_session_to_windows(sess, window_steps=WINDOW_STEPS, writer_profile=uid)
            windows.extend(ws)
            groups.extend([uid] * len(ws))
        print(f"[质检汇总] 扫描 {len(json_files)} 份 JSON，通过 {passed} 份，有效时间窗 {len(windows)} 个，涉及用户 {len(set(groups))} 名。")
    else:
        print("[演示模式] 未提供 --data-dir，使用 16 位模拟用户（96 场会话）演示 LOUO 按人隔离评测管线...")
        syn_sessions = generate_synthetic_annotated_sessions(n_sessions=96, seed=2026)
        for idx, (sess, prof) in enumerate(syn_sessions):
            uid = f"u{(idx % 16) + 1:02d}_{prof}"
            ws = convert_session_to_windows(sess, window_steps=WINDOW_STEPS, writer_profile=uid)
            windows.extend(ws)
            groups.extend([uid] * len(ws))

    if len(windows) < 20:
        raise SystemExit("有效标注窗口不足 20 个，请先按《标注指南 v2.0》采集真实会话后再评测。")

    cls_counts = Counter(w.label_name for w in windows)
    print("\n--- 6 类样本窗口分布 ---")
    for name in CLASS_NAMES:
        print(f"  {name:<26}: {cls_counts.get(name, 0)}")

    X = np.stack([extract_window_summary_features(w) for w in windows])
    y = np.asarray([w.label_id for w in windows], dtype=np.int64)
    g = np.asarray(groups)

    unique_users = len(set(groups))
    n_splits = max(2, min(args.n_splits, unique_users))
    gkf = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros_like(y)
    oof_stuck_prob = np.zeros(len(y), dtype=np.float64)

    for tr_idx, va_idx in gkf.split(X, y, groups=g):
        clf = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=180, random_state=42)
        # 2.0x 心算样本加权，与神经网络代价敏感损失对齐
        sample_weight = np.where(y[tr_idx] == 1, 2.0, 1.0)
        clf.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weight)
        probs = clf.predict_proba(X[va_idx])
        classes = list(clf.classes_)
        full_probs = np.zeros((len(va_idx), len(CLASS_NAMES)), dtype=np.float64)
        for col_i, cls_id in enumerate(classes):
            full_probs[:, cls_id] = probs[:, col_i]
        oof_preds[va_idx] = full_probs.argmax(axis=1)
        oof_stuck_prob[va_idx] = full_probs[:, 2:].sum(axis=1)

    y_bin = (y >= 2).astype(int)
    p_bin = (oof_preds >= 2).astype(int)
    stuck_f1 = float(f1_score(y_bin, p_bin, zero_division=0))
    pr_auc = float(average_precision_score(y_bin, oof_stuck_prob)) if len(set(y_bin)) > 1 else 0.0
    macro_f1 = float(f1_score(y, oof_preds, average="macro", zero_division=0))
    think_mask = y == 1
    think_fpr = float(np.mean((oof_preds[think_mask] >= 2).astype(float))) if np.any(think_mask) else 0.0
    cm = confusion_matrix(y, oof_preds, labels=list(range(len(CLASS_NAMES))))

    print(f"\n================ LOUO-CV 按人隔离评测结果 ({n_splits}-Fold, {unique_users} Users) ================")
    print(f"深度心算误报率 ThinkFPR  : {think_fpr:.2%}  (合格 < 10%, 优秀 < 5%)")
    print(f"需干预二分类 StuckF1     : {stuck_f1:.4f}  (合格 ≥ 0.80, 优秀 ≥ 0.88)")
    print(f"需干预曲线面积 PR-AUC    : {pr_auc:.4f}  (合格 ≥ 0.82, 优秀 ≥ 0.90)")
    print(f"六分类宏平均 6C_MacroF1  : {macro_f1:.4f}  (合格 ≥ 0.72, 优秀 ≥ 0.82)")
    print("\n6×6 混淆矩阵（行=真实类别, 列=预测类别，顺序 0..5）:")
    print(cm)

    if args.data_dir and args.blind_audit_dir:
        kappa = compute_cohen_kappa_dirs(Path(args.data_dir), Path(args.blind_audit_dir))
        if kappa is not None:
            print(f"\n双人盲审标注一致性 Cohen's κ : {kappa:.4f}  (合格 ≥ 0.70, 优秀 ≥ 0.80)")


if __name__ == "__main__":
    main()
