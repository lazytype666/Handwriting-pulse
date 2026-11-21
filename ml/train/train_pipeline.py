"""
《笔迹脉搏》RTX 5060 GPU 四阶段解耦训练与联合评估管线。

阶段划分：
  Stage 1 (`spatial_pretrain`):
      在无时间 6 通道等弧长空间流上使用 CTCLoss 训练 [模型 1] EdgeMathEncoder
      （支持直接读取官方 Google MathWriting 2024 .tgz + 大学数学高阶符号增强）。
  Stage 2 (`temporal_ssl`):
      基于“笔画决策步级 10 维动力学流”对严格单向 CausalDynamicsEncoder 执行自监督节奏预训练。
  Stage 3 (`cognitive_finetune`):
      冻结模型 1 主干提取各笔画步的语义特征 Z_sem[j]，与动力学流做严格因果 Cross-Attention，
      配合代价敏感损失（2.0× 心算误报惩罚）训练 6 分类 [模型 2] CognitiveStateNet。
  Stage 4 (`evaluate`):
      全链路评测 CTC 解码词错率、6 类认知状态准确率、Macro-F1 与深度心算防误报率。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data_pipeline.mathwriting_loader import (  # noqa: E402
    MAX_SPATIAL_LEN,
    generate_university_math_sequences,
    load_mathwriting_subset_from_dir,
    load_mathwriting_subset_from_tgz,
)
from ml.data_pipeline.schema_converter import (  # noqa: E402
    CLASS_NAMES,
    STROKE_MAX_PTS,
    WINDOW_STEPS,
    StrokeWindowSample,
    convert_session_to_windows,
    generate_synthetic_annotated_sessions,
    load_inkpulse_json,
)
from ml.data_pipeline.vocab import VOCAB_SIZE, decode_ctc_beam_search, decode_ctc_greedy  # noqa: E402
from ml.models.decoupled_dual_stream import MODEL_CONFIG_DEFAULTS, TORCH_AVAILABLE  # noqa: E402


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new_dp = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            new_dp.append(min(dp[j] + 1, new_dp[-1] + 1, dp[j - 1] + cost))
        dp = new_dp
    return dp[-1]


def collect_cognitive_windows(data_dir: str | None, n_syn_sessions: int = 120) -> List[StrokeWindowSample]:
    samples: List[StrokeWindowSample] = []
    if data_dir and Path(data_dir).exists():
        for jf in sorted(Path(data_dir).glob("*.json")):
            samples.extend(convert_session_to_windows(load_inkpulse_json(jf), writer_profile="real_user"))
    if not samples:
        for sess_obj, prof in generate_synthetic_annotated_sessions(n_sessions=n_syn_sessions, seed=2026):
            samples.extend(convert_session_to_windows(sess_obj, writer_profile=prof))
    return samples


def _augment_spatial_feature(feat: np.ndarray) -> np.ndarray:
    """
    在线抗过拟合几何与时序增强（仅作用于训练集）：
    1. 随机各向异性缩放 [0.88, 1.12] + 随机轻微旋转/斜体 [-7°, +7°]，并重新归一化方向余弦；
    2. 随机轻度垂向基线扰动；
    3. 轻度 SpecAugment 连续短帧遮蔽（18% 概率遮蔽 2~4 帧），防止死记硬背单一笔锋路径。
    """
    x = feat.copy()
    seq_len = len(x)
    if seq_len < 4:
        return x

    sx = float(np.random.uniform(0.88, 1.12))
    sy = float(np.random.uniform(0.88, 1.12))
    theta = float(np.random.uniform(-0.122, 0.122))  # ±7 deg
    cos_a = float(np.cos(theta))
    sin_a = float(np.sin(theta))

    dx = x[:, 0] * sx
    dy = x[:, 1] * sy
    rdx = np.clip(dx * cos_a - dy * sin_a, -4.0, 4.0)
    rdy = np.clip(dx * sin_a + dy * cos_a, -4.0, 4.0)
    ds = np.hypot(rdx, rdy)
    valid = ds > 1e-6
    x[:, 0] = rdx
    x[:, 1] = rdy
    x[:, 2] = np.where(valid, rdx / np.where(valid, ds, 1.0), 1.0)
    x[:, 3] = np.where(valid, rdy / np.where(valid, ds, 1.0), 0.0)
    x[:, 5] = np.clip(x[:, 5] * sy + float(np.random.uniform(-0.05, 0.05)), -1.5, 1.5)

    # 轻度帧段随机遮蔽（SpecAugment）
    if seq_len >= 16 and np.random.random() < 0.18:
        span = int(np.random.randint(2, min(5, max(3, seq_len // 10))))
        start = int(np.random.randint(1, seq_len - span))
        x[start : start + span, 0] = 0.0
        x[start : start + span, 1] = 0.0
        x[start : start + span, 2] = 1.0
        x[start : start + span, 3] = 0.0
        x[start : start + span, 5] = 0.0
    return x


def _eval_ctc_cer(
    model,
    loader,
    device,
    use_beam: bool = False,
    beam_width: int = 4,
) -> Tuple[float, str]:
    import torch

    model.eval()
    total_ed = 0
    total_chars = 0
    sample_preview = ""
    with torch.no_grad():
        for x, pmask, in_lens, tgts, tgt_lens, _ in loader:
            x, pmask = x.to(device, non_blocking=True), pmask.to(device, non_blocking=True)
            _, _, logits = model(x, pad_mask=pmask)
            logits_cpu = logits.float().cpu()
            preds = logits_cpu.argmax(dim=-1).tolist()
            offset = 0
            for b_idx, t_len in enumerate(tgt_lens.tolist()):
                sub_l = int(in_lens[b_idx].item())
                if use_beam:
                    pred_str = decode_ctc_beam_search(
                        logits_cpu[b_idx, :sub_l].numpy(), beam_width=beam_width, top_k=6
                    ).replace(" ", "")
                else:
                    pred_str = decode_ctc_greedy(preds[b_idx][:sub_l]).replace(" ", "")
                gt_ids = tgts[offset : offset + t_len].tolist()
                gt_str = decode_ctc_greedy(gt_ids).replace(" ", "")
                offset += t_len
                total_ed += _edit_distance(pred_str, gt_str)
                total_chars += max(1, len(gt_str))
                if not sample_preview:
                    sample_preview = f"pred='{pred_str}' vs gt='{gt_str}'"
    return total_ed / max(1, total_chars), sample_preview


class _SpatialCTCDataset:
    def __init__(self, items: List[Tuple[np.ndarray, List[int], str]], is_train: bool = False) -> None:
        self.items = items
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        feat, tok_ids, raw_latex = self.items[idx]
        if self.is_train:
            feat = _augment_spatial_feature(feat)
        return feat, np.asarray(tok_ids, dtype=np.int64), raw_latex


def _collate_ctc_batch(batch):
    import torch

    max_l = min(MAX_SPATIAL_LEN, max(len(x[0]) for x in batch))
    max_l = max(8, max_l + (max_l % 2))
    x_pad = torch.zeros(len(batch), max_l, 6, dtype=torch.float32)
    pad_mask = torch.ones(len(batch), max_l, dtype=torch.bool)
    in_lens = []
    targets = []
    tgt_lens = []
    raw_texts = []
    for b_idx, (feat, tok_ids, raw_latex) in enumerate(batch):
        l_cur = min(max_l, len(feat))
        x_pad[b_idx, :l_cur] = torch.from_numpy(feat[:l_cur])
        pad_mask[b_idx, :l_cur] = False
        sub_len = max(1, (l_cur + 1) // 2)
        in_lens.append(sub_len)
        targets.append(torch.from_numpy(tok_ids))
        tgt_lens.append(len(tok_ids))
        raw_texts.append(raw_latex)
    return (
        x_pad,
        pad_mask,
        torch.tensor(in_lens, dtype=torch.long),
        torch.cat(targets, dim=0),
        torch.tensor(tgt_lens, dtype=torch.long),
        raw_texts,
    )


def train_spatial_ctc(args: argparse.Namespace) -> None:
    import math
    import time
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from ml.models.decoupled_dual_stream import EdgeMathEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_pairs: List[Tuple[np.ndarray, List[int], str]] = []
    real_val_pairs: List[Tuple[np.ndarray, List[int], str]] = []
    real_test_pairs: List[Tuple[np.ndarray, List[int], str]] = []
    mw_target = args.mathwriting_path or args.mathwriting_tgz
    if mw_target and Path(mw_target).exists():
        p_mw = Path(mw_target)
        if p_mw.is_dir():
            print(f"[Stage 1] 正在并行加载 MathWriting (symbols+train, max_len={MAX_SPATIAL_LEN}): {p_mw}")
            t_load0 = time.perf_counter()
            train_pairs.extend(
                load_mathwriting_subset_from_dir(
                    p_mw,
                    splits=("symbols", "train"),
                    max_samples=args.mw_max_samples,
                    max_seq_len=MAX_SPATIAL_LEN,
                    shuffle_seed=42,
                )
            )
            real_val_pairs.extend(
                load_mathwriting_subset_from_dir(
                    p_mw,
                    splits=("valid",),
                    max_samples=min(5000, max(1500, args.mw_max_samples // 20)),
                    max_seq_len=MAX_SPATIAL_LEN,
                    shuffle_seed=101,
                )
            )
            real_test_pairs.extend(
                load_mathwriting_subset_from_dir(
                    p_mw,
                    splits=("test",),
                    max_samples=4000,
                    max_seq_len=MAX_SPATIAL_LEN,
                    shuffle_seed=202,
                )
            )
            print(
                f"[Stage 1] 加载耗时 {time.perf_counter()-t_load0:.1f}s | "
                f"真实训练={len(train_pairs)} | 独立真实验证={len(real_val_pairs)} | 留出真实测试={len(real_test_pairs)}"
            )
        else:
            print(f"[Stage 1] 正在流式解析 MathWriting 数据包: {p_mw}")
            mw_samples = load_mathwriting_subset_from_tgz(
                p_mw, split_prefix="train/", max_samples=args.mw_max_samples, max_seq_len=MAX_SPATIAL_LEN
            )
            print(f"[Stage 1] 成功载入 {len(mw_samples)} 条真实 MathWriting 手写公式。")
            train_pairs.extend(mw_samples)

    aug_samples = generate_university_math_sequences(n_samples=args.aug_samples, seed=2026)
    print(f"[Stage 1] 混入大学数学全符号增强集: {len(aug_samples)} 条（占比 {100.0*len(aug_samples)/max(1,len(train_pairs)+len(aug_samples)):.1f}% 防合成过拟合）。")
    aug_split = max(1, int(len(aug_samples) * 0.9))
    univ_val_pairs = aug_samples[aug_split:]
    train_pairs.extend(aug_samples[:aug_split])
    if not real_val_pairs:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(train_pairs))
        s_idx = max(1, int(len(train_pairs) * 0.9))
        real_val_pairs = [train_pairs[i] for i in perm[s_idx:]]
        train_pairs = [train_pairs[i] for i in perm[:s_idx]]

    # 固定 1500 条训练探针，用于实时比对 train_CER vs real_val_CER 监控过拟合
    probe_rng = np.random.default_rng(7)
    probe_idx = probe_rng.choice(len(train_pairs), size=min(1500, len(train_pairs)), replace=False)
    train_probe_pairs = [train_pairs[int(i)] for i in probe_idx]

    n_workers = 4 if device.type == "cuda" else 0
    tr_loader = DataLoader(
        _SpatialCTCDataset(train_pairs, is_train=True),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_ctc_batch,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
    )
    tr_probe_loader = DataLoader(
        _SpatialCTCDataset(train_probe_pairs, is_train=False),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_ctc_batch,
    )
    va_loader = DataLoader(
        _SpatialCTCDataset(real_val_pairs, is_train=False),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_ctc_batch,
    )
    univ_va_loader = DataLoader(
        _SpatialCTCDataset(univ_val_pairs, is_train=False),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_ctc_batch,
    )
    test_loader = (
        DataLoader(
            _SpatialCTCDataset(real_test_pairs, is_train=False),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate_ctc_batch,
        )
        if real_test_pairs
        else None
    )

    math_d_model = int(MODEL_CONFIG_DEFAULTS.get("math_d_model", MODEL_CONFIG_DEFAULTS["d_model"]))
    model = EdgeMathEncoder(
        in_dim=int(MODEL_CONFIG_DEFAULTS["spatial_in_dim"]),
        d_model=math_d_model,
        n_heads=int(MODEL_CONFIG_DEFAULTS["n_heads"]),
        num_layers=int(MODEL_CONFIG_DEFAULTS["conformer_layers"]),
        math_vocab_size=VOCAB_SIZE,
        dropout=float(MODEL_CONFIG_DEFAULTS["dropout"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    warmup_epochs = min(3, max(1, args.epochs // 12))

    def lr_lambda(ep_idx: int) -> float:
        if ep_idx < warmup_epochs:
            return float(ep_idx + 1) / float(warmup_epochs)
        progress = float(ep_idx - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ctc_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    best_cer = 999.0
    out_path = save_dir / "stage1_edge_math_encoder.pt"
    print(
        f"[Stage 1] 设备={device} (bf16={use_bf16}) | EdgeMathEncoder(d={math_d_model}, L={MODEL_CONFIG_DEFAULTS['conformer_layers']}, "
        f"V={VOCAB_SIZE}, params={n_params/1e3:.1f}K) 开始抗过拟合增强训练..."
    )

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss = 0.0
        for x, pmask, in_lens, tgts, tgt_lens, _ in tr_loader:
            x = x.to(device, non_blocking=True)
            pmask = pmask.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                _, _, logits = model(x, pad_mask=pmask)
            # CTCLoss 始终保持 FP32 确保数值稳定
            log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
            loss = ctc_fn(log_probs, tgts, in_lens, tgt_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            tr_loss += float(loss.item())
        scheduler.step()

        real_val_cer, sample_preview = _eval_ctc_cer(model, va_loader, device, use_beam=False)
        if real_val_cer <= best_cer:
            best_cer = real_val_cer
            torch.save(model.state_dict(), out_path)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            tr_probe_cer, _ = _eval_ctc_cer(model, tr_probe_loader, device, use_beam=False)
            univ_cer, _ = _eval_ctc_cer(model, univ_va_loader, device, use_beam=False)
            gap = real_val_cer - tr_probe_cer
            print(
                f"[Stage 1] Epoch {epoch:02d}/{args.epochs} ({time.perf_counter()-ep_t0:.1f}s) | "
                f"loss={tr_loss/max(1,len(tr_loader)):.4f} | train_CER={tr_probe_cer:.4f} | "
                f"real_val_CER={real_val_cer:.4f} (gap={gap:+.4f}) | univ_CER={univ_cer:.4f} | {sample_preview}"
            )

    # 加载最优权重并在完全未见过的真实 MathWriting test 集上比对 Greedy vs Prefix Beam Search
    model.load_state_dict(torch.load(out_path, map_location=device), strict=True)
    print(f"[Stage 1 完成] 最优权重保存于：{out_path} (best_real_val_CER={best_cer:.4f})")
    if test_loader is not None:
        test_greedy_cer, preview_g = _eval_ctc_cer(model, test_loader, device, use_beam=False)
        test_beam_cer, preview_b = _eval_ctc_cer(model, test_loader, device, use_beam=True, beam_width=4)
        print(
            f"[Stage 1 留出测试集评测 (n={len(real_test_pairs)})] "
            f"real_test_Greedy_CER={test_greedy_cer:.4f} | real_test_Beam4_CER={test_beam_cer:.4f} | {preview_b}"
        )


def train_temporal_ssl(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from ml.models.decoupled_dual_stream import CausalDynamicsEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    windows = collect_cognitive_windows(args.data_dir, n_syn_sessions=120)
    dyn_tensor = torch.tensor(np.stack([w.dynamics_seq for w in windows]), dtype=torch.float32)
    loader = DataLoader(TensorDataset(dyn_tensor), batch_size=args.batch_size, shuffle=True)

    enc = CausalDynamicsEncoder(
        in_dim=int(MODEL_CONFIG_DEFAULTS["dynamics_in_dim"]),
        d_model=int(MODEL_CONFIG_DEFAULTS["d_model"]),
        num_layers=int(MODEL_CONFIG_DEFAULTS["gru_layers"]),
        dropout=float(MODEL_CONFIG_DEFAULTS["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-4)

    out_path = save_dir / "stage2_causal_dynamics.pt"
    print(f"[Stage 2] 使用设备 {device} 预训练因果节奏自监督编码器...")
    for epoch in range(1, args.epochs + 1):
        enc.train()
        total_loss = 0.0
        for (dy,) in loader:
            dy = dy.to(device)
            # 严格因果 Next-Step 预测：用前 t 步预测 t+1 步动力学脉冲特征
            _, pred_next = enc(dy[:, :-1, :])
            loss = F.mse_loss(pred_next, dy[:, 1:, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"[Stage 2 SSL] Epoch {epoch:02d}/{args.epochs} | next_step_mse={total_loss/max(1,len(loader)):.4f}")
    torch.save(enc.state_dict(), out_path)
    print(f"[Stage 2 完成] 自监督节奏权重保存于：{out_path}")


def _extract_step_semantic_embeddings(math_encoder, stroke_sp_tensor: "torch.Tensor", device) -> "torch.Tensor":
    """用冻结的 EdgeMathEncoder 将 [B, W, L_s, 6] 每笔几何序列编码为 [B, W, D_math] 语义序列（空步置零防 NaN）。"""
    import torch

    bsz, w_steps, l_s, c_in = stroke_sp_tensor.shape
    flat = stroke_sp_tensor.view(bsz * w_steps, l_s, c_in).to(device)
    non_empty_frames = flat.abs().sum(dim=-1) > 1e-6  # [B*W, L_s]
    valid_stroke_mask = non_empty_frames.any(dim=-1)  # [B*W]
    d_model = int(MODEL_CONFIG_DEFAULTS.get("math_d_model", MODEL_CONFIG_DEFAULTS["d_model"]))
    out_emb = torch.zeros(bsz * w_steps, d_model, device=device, dtype=torch.float32)

    if valid_stroke_mask.any():
        sub_flat = flat[valid_stroke_mask]
        sub_pad = ~non_empty_frames[valid_stroke_mask]
        with torch.no_grad():
            _, pooled, _ = math_encoder(sub_flat, pad_mask=sub_pad)
        out_emb[valid_stroke_mask] = pooled

    return out_emb.view(bsz, w_steps, d_model)


def train_cognitive_finetune(args: argparse.Namespace) -> None:
    import torch
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader, TensorDataset
    from ml.models.decoupled_dual_stream import (
        CognitiveStateNet,
        EdgeMathEncoder,
        cognitive_cost_sensitive_loss,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    math_ckpt = Path(args.math_checkpoint or (save_dir / "stage1_edge_math_encoder.pt"))
    dyn_ckpt = Path(args.dyn_checkpoint or (save_dir / "stage2_causal_dynamics.pt"))

    math_d_model = int(MODEL_CONFIG_DEFAULTS.get("math_d_model", MODEL_CONFIG_DEFAULTS["d_model"]))
    math_enc = EdgeMathEncoder(
        in_dim=int(MODEL_CONFIG_DEFAULTS["spatial_in_dim"]),
        d_model=math_d_model,
        n_heads=int(MODEL_CONFIG_DEFAULTS["n_heads"]),
        num_layers=int(MODEL_CONFIG_DEFAULTS["conformer_layers"]),
        math_vocab_size=VOCAB_SIZE,
        dropout=0.0,
    ).to(device)
    if math_ckpt.exists():
        math_enc.load_state_dict(torch.load(math_ckpt, map_location=device), strict=True)
        print(f"[Stage 3] 加载已训练空间编码器：{math_ckpt}")
    math_enc.eval()
    for p in math_enc.parameters():
        p.requires_grad = False

    cog_net = CognitiveStateNet(MODEL_CONFIG_DEFAULTS).to(device)
    if dyn_ckpt.exists():
        cog_net.dynamics_encoder.load_state_dict(torch.load(dyn_ckpt, map_location=device), strict=True)
        print(f"[Stage 3] 加载已预训练因果动力学权重：{dyn_ckpt}")

    windows = collect_cognitive_windows(args.data_dir, n_syn_sessions=240)
    sp_steps = torch.tensor(np.stack([w.stroke_spatial_seqs for w in windows]), dtype=torch.float32)
    dy_steps = torch.tensor(np.stack([w.dynamics_seq for w in windows]), dtype=torch.float32)
    y_tensor = torch.tensor([w.label_id for w in windows], dtype=torch.long)
    mra_tensor = torch.tensor([[w.latent_m, w.latent_r, w.latent_a] for w in windows], dtype=torch.float32)

    # 先批量预抽取各笔画步的冻结语义向量 Z_sem
    sem_list = []
    for start in range(0, len(windows), 64):
        sem_chunk = _extract_step_semantic_embeddings(math_enc, sp_steps[start : start + 64], device).cpu()
        sem_list.append(sem_chunk)
    sem_tensor = torch.cat(sem_list, dim=0)

    full_ds = TensorDataset(dy_steps, sem_tensor, y_tensor, mra_tensor)
    n_val = max(12, int(len(full_ds) * 0.2))
    n_tr = len(full_ds) - n_val
    tr_ds, va_ds = torch.utils.data.random_split(full_ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(cog_net.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score = -1.0
    out_path = save_dir / "stage3_cognitive_state_net.pt"

    print(f"[Stage 3] 使用设备 {device} 微调 6 分类 CognitiveStateNet (n_train={n_tr}, n_val={n_val})...")
    for epoch in range(1, args.epochs + 1):
        cog_net.train()
        tr_loss = 0.0
        for dy, sem, y, mra in tr_loader:
            dy, sem, y, mra = dy.to(device), sem.to(device), y.to(device), mra.to(device)
            logits, surp, pred_mra = cog_net(dy, sem)
            loss = cognitive_cost_sensitive_loss(logits, y, surp, pred_mra, mra)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cog_net.parameters(), 2.0)
            optimizer.step()
            tr_loss += float(loss.item())

        cog_net.eval()
        all_preds: List[int] = []
        all_y: List[int] = []
        with torch.no_grad():
            for dy, sem, y, _ in va_loader:
                dy, sem = dy.to(device), sem.to(device)
                logits, _, _ = cog_net(dy, sem)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_y.extend(y.tolist())

        y_arr = np.asarray(all_y)
        p_arr = np.asarray(all_preds)
        acc_6c = float(np.mean(p_arr == y_arr))
        macro_f1 = float(f1_score(y_arr, p_arr, average="macro", zero_division=0))
        stuck_f1 = float(f1_score((y_arr >= 2).astype(int), (p_arr >= 2).astype(int), zero_division=0))
        think_mask = y_arr == 1
        think_fpr = float(np.mean((p_arr[think_mask] >= 2).astype(float))) if np.any(think_mask) else 0.0

        # 综合打分优先严控心算误报率
        score = stuck_f1 - 0.5 * think_fpr + 0.3 * macro_f1
        if score >= best_score:
            best_score = score
            torch.save(cog_net.state_dict(), out_path)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"[Stage 3] Epoch {epoch:02d}/{args.epochs} | loss={tr_loss/max(1,len(tr_loader)):.4f} | "
                f"6C_Acc={acc_6c:.4f} | 6C_MacroF1={macro_f1:.4f} | StuckF1={stuck_f1:.4f} | ThinkFPR={think_fpr:.2%}"
            )
    print(f"[Stage 3 完成] 最优 CognitiveStateNet 权重已保存：{out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="《笔迹脉搏》PyTorch GPU 四阶段解耦训练脚本")
    parser.add_argument(
        "--stage",
        choices=["spatial_pretrain", "temporal_ssl", "cognitive_finetune", "all"],
        default="all",
    )
    parser.add_argument("--mathwriting-path", type=str, default=None, help="已解压的 MathWriting 目录或 .tgz 路径")
    parser.add_argument("--mathwriting-tgz", type=str, default=None, help="MathWriting .tgz 数据路径（兼容旧参数）")
    parser.add_argument("--mw-max-samples", type=int, default=20000)
    parser.add_argument("--aug-samples", type=int, default=3000)
    parser.add_argument("--data-dir", type=str, default=None, help="App 导出的 v2.0-annotated JSON 目录")
    parser.add_argument("--save-dir", type=str, default="ml/runs")
    parser.add_argument("--math-checkpoint", type=str, default=None)
    parser.add_argument("--dyn-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit("当前 Python 环境未安装 PyTorch。请在 WSL2 CUDA 环境中使用 /root/.inkpulse_venv/bin/python 运行。")

    if args.stage in ("spatial_pretrain", "all"):
        train_spatial_ctc(args)
    if args.stage in ("temporal_ssl", "all"):
        train_temporal_ssl(args)
    if args.stage in ("cognitive_finetune", "all"):
        train_cognitive_finetune(args)


if __name__ == "__main__":
    main()
