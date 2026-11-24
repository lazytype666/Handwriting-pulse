"""
《笔迹脉搏》RTX 5060 GPU 四阶段解耦 + 笔画关系图联合训练管线（Quote ③ + Quote ④ + 图四升级版）。

阶段划分：
  Stage 1 (`spatial_pretrain`):
      在 10 通道等弧长空间流上联合训练 [模型 1] EdgeMathGraphNet：
        - 主任务：Google MathWriting + 大学数学高阶表达式 CTCLoss
        - 图任务：依托 MathWriting `symbols/` 拼装的 2D 布局图真值训练双线性 `StrokeGraphEdgeHead` (7 类边关系)
        - 可选 (`--train-coverage`)：同时微调 CoMER 风格 `CoverageMathSeq2SeqDecoder` 供 ONNX 原生加速轨
  Stage 2 (`temporal_ssl`):
      基于 14 维（动力学 10 维 + 图谱对照 4 维）决策步流对严格单向 CausalDynamicsEncoder 做自监督预测训练。
  Stage 3 (`cognitive_finetune`):
      冻结模型 1 提取各笔画 96 维语义嵌入，与 14 维因果动力学+图谱流做 Cross-Attention，
      配合代价敏感损失（2.0× 心算误报惩罚）训练 6 分类 CognitiveStateNet。
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
    SPATIAL_DIM,
    StrokeGraphSample,
    generate_stroke_graph_samples,
    generate_university_math_sequences,
    load_mathwriting_subset_from_dir,
    load_mathwriting_subset_from_tgz,
    load_symbol_bank_from_dir,
)
from ml.data_pipeline.schema_converter import (  # noqa: E402
    CLASS_NAMES,
    DYNAMICS_DIM,
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
    在线抗过拟合 10 通道几何增强（同步更新 2D 包围盒几何通道 6..9）：
    1. 各向异性缩放 [0.88, 1.12] + 随机轻微旋转/斜体 [-7°, +7°]；
    2. 轻度 SpecAugment 短帧遮蔽（18% 概率遮蔽 2~4 帧）。
    """
    x = feat.copy()
    seq_len = len(x)
    if seq_len < 4:
        return x

    sx = float(np.random.uniform(0.88, 1.12))
    sy = float(np.random.uniform(0.88, 1.12))
    theta = float(np.random.uniform(-0.122, 0.122))
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
    x[:, 5] = np.clip(x[:, 5] * sy + float(np.random.uniform(-0.04, 0.04)), -1.5, 1.5)

    if x.shape[1] >= 10:
        cx_s = x[:, 6] * sx
        cy_s = x[:, 7] * sy
        x[:, 6] = np.clip(cx_s * cos_a - cy_s * sin_a, -2.5, 2.5)
        x[:, 7] = np.clip(cx_s * sin_a + cy_s * cos_a, -1.5, 1.5)
        x[:, 8] = np.clip(x[:, 8] * sx, 0.0, 3.0)
        x[:, 9] = np.clip(x[:, 9] * sy, 0.0, 3.0)

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
    enforce_grammar: bool = False,
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
                        logits_cpu[b_idx, :sub_l].numpy(),
                        beam_width=beam_width,
                        top_k=6,
                        enforce_grammar=enforce_grammar,
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


def _eval_edge_acc(model, graph_samples: List[StrokeGraphSample], device) -> float:
    import torch

    if not graph_samples:
        return 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for g in graph_samples:
            sp = torch.from_numpy(g.stroke_seqs).to(device)
            non_empty = sp.abs().sum(dim=-1) > 1e-6
            pmask = ~non_empty
            _, sem_strokes, _ = model(sp, pad_mask=pmask)
            idx_t = torch.from_numpy(g.pair_indices).to(device)
            geom_t = torch.from_numpy(g.pair_geom).to(device)
            labels_t = torch.from_numpy(g.pair_labels).to(device)
            logits = model.forward_edge_head(sem_strokes[idx_t[:, 0]], sem_strokes[idx_t[:, 1]], geom_t)
            pred = logits.argmax(dim=-1)
            correct += int((pred == labels_t).sum().item())
            total += int(labels_t.numel())
    return correct / max(1, total)


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

    in_dim = int(batch[0][0].shape[1]) if len(batch[0][0].shape) == 2 else SPATIAL_DIM
    max_l = min(MAX_SPATIAL_LEN, max(len(x[0]) for x in batch))
    max_l = max(8, max_l + (max_l % 2))
    x_pad = torch.zeros(len(batch), max_l, in_dim, dtype=torch.float32)
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
    from ml.models.decoupled_dual_stream import CoverageMathSeq2SeqDecoder, EdgeMathGraphNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    spatial_in_dim = int(MODEL_CONFIG_DEFAULTS["spatial_in_dim"])

    train_pairs: List[Tuple[np.ndarray, List[int], str]] = []
    real_val_pairs: List[Tuple[np.ndarray, List[int], str]] = []
    real_test_pairs: List[Tuple[np.ndarray, List[int], str]] = []
    symbol_bank = {}

    mw_target = args.mathwriting_path or args.mathwriting_tgz
    if mw_target and Path(mw_target).exists():
        p_mw = Path(mw_target)
        if p_mw.is_dir():
            print(f"[Stage 1] 正在并行加载 10 通道 MathWriting (symbols+train, max_len={MAX_SPATIAL_LEN}): {p_mw}")
            t_load0 = time.perf_counter()
            symbol_bank = load_symbol_bank_from_dir(p_mw, max_per_symbol=80)
            train_pairs.extend(
                load_mathwriting_subset_from_dir(
                    p_mw,
                    splits=("symbols", "train"),
                    max_samples=args.mw_max_samples,
                    max_seq_len=MAX_SPATIAL_LEN,
                    shuffle_seed=42,
                    spatial_dim=spatial_in_dim,
                )
            )
            real_val_pairs.extend(
                load_mathwriting_subset_from_dir(
                    p_mw,
                    splits=("valid",),
                    max_samples=min(5000, max(1500, args.mw_max_samples // 20)),
                    max_seq_len=MAX_SPATIAL_LEN,
                    shuffle_seed=101,
                    spatial_dim=spatial_in_dim,
                )
            )
            real_test_pairs.extend(
                load_mathwriting_subset_from_dir(
                    p_mw,
                    splits=("test",),
                    max_samples=4000,
                    max_seq_len=MAX_SPATIAL_LEN,
                    shuffle_seed=202,
                    spatial_dim=spatial_in_dim,
                )
            )
            print(
                f"[Stage 1] 加载耗时 {time.perf_counter()-t_load0:.1f}s | symbols银行={len(symbol_bank)}类 | "
                f"训练={len(train_pairs)} | 验证={len(real_val_pairs)} | 留出测试={len(real_test_pairs)}"
            )
        else:
            mw_samples = load_mathwriting_subset_from_tgz(
                p_mw, split_prefix="train/", max_samples=args.mw_max_samples, max_seq_len=MAX_SPATIAL_LEN, spatial_dim=spatial_in_dim
            )
            train_pairs.extend(mw_samples)

    aug_samples = generate_university_math_sequences(n_samples=args.aug_samples, seed=2026, spatial_dim=spatial_in_dim)
    aug_split = max(1, int(len(aug_samples) * 0.9))
    univ_val_pairs = aug_samples[aug_split:]
    train_pairs.extend(aug_samples[:aug_split])
    if not real_val_pairs:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(train_pairs))
        s_idx = max(1, int(len(train_pairs) * 0.9))
        real_val_pairs = [train_pairs[i] for i in perm[s_idx:]]
        train_pairs = [train_pairs[i] for i in perm[:s_idx]]

    # 2D 笔画图关系训练集与验证集（真人类手写符号库拼装 7 类空间关系）
    graph_all = generate_stroke_graph_samples(
        n_samples=args.graph_samples,
        seed=2026,
        symbol_bank=symbol_bank,
        stroke_seq_len=24,
        spatial_dim=spatial_in_dim,
    )
    g_split = max(1, int(len(graph_all) * 0.85))
    graph_train = graph_all[:g_split]
    graph_val = graph_all[g_split:]
    print(f"[Stage 1] 2D 笔画图关系样本数: train={len(graph_train)}, val={len(graph_val)}")

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
    tr_probe_loader = DataLoader(_SpatialCTCDataset(train_probe_pairs, is_train=False), batch_size=args.batch_size, shuffle=False, collate_fn=_collate_ctc_batch)
    va_loader = DataLoader(_SpatialCTCDataset(real_val_pairs, is_train=False), batch_size=args.batch_size, shuffle=False, collate_fn=_collate_ctc_batch)
    univ_va_loader = DataLoader(_SpatialCTCDataset(univ_val_pairs, is_train=False), batch_size=args.batch_size, shuffle=False, collate_fn=_collate_ctc_batch)
    test_loader = DataLoader(_SpatialCTCDataset(real_test_pairs, is_train=False), batch_size=args.batch_size, shuffle=False, collate_fn=_collate_ctc_batch) if real_test_pairs else None

    math_d_model = int(MODEL_CONFIG_DEFAULTS["math_d_model"])
    model = EdgeMathGraphNet(
        in_dim=spatial_in_dim,
        d_model=math_d_model,
        n_heads=int(MODEL_CONFIG_DEFAULTS["n_heads"]),
        num_layers=int(MODEL_CONFIG_DEFAULTS["conformer_layers"]),
        math_vocab_size=VOCAB_SIZE,
        edge_geom_dim=int(MODEL_CONFIG_DEFAULTS["edge_geom_dim"]),
        edge_hidden_dim=int(MODEL_CONFIG_DEFAULTS["edge_hidden_dim"]),
        edge_classes=int(MODEL_CONFIG_DEFAULTS["edge_classes"]),
        dropout=float(MODEL_CONFIG_DEFAULTS["dropout"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    edge_params = sum(p.numel() for p in model.edge_head.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    warmup_epochs = min(3, max(1, args.epochs // 12))

    def lr_lambda(ep_idx: int) -> float:
        if ep_idx < warmup_epochs:
            return float(ep_idx + 1) / float(warmup_epochs)
        progress = float(ep_idx - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ctc_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    best_score = 999.0
    out_path = save_dir / "stage1_edge_math_encoder.pt"
    print(
        f"[Stage 1] 设备={device} (bf16={use_bf16}) | EdgeMathGraphNet(in={spatial_in_dim}, d={math_d_model}, "
        f"L={MODEL_CONFIG_DEFAULTS['conformer_layers']}, total_params={n_params/1e3:.1f}K, edge_head={edge_params/1e3:.1f}K)"
    )

    graph_rng = np.random.default_rng(123)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_ctc_loss = 0.0
        tr_edge_loss = 0.0
        edge_steps = 0

        for step_idx, (x, pmask, in_lens, tgts, tgt_lens, _) in enumerate(tr_loader):
            x = x.to(device, non_blocking=True)
            pmask = pmask.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                _, _, logits = model(x, pad_mask=pmask)
            loss_ctc = ctc_fn(F.log_softmax(logits.float(), dim=-1).transpose(0, 1), tgts, in_lens, tgt_lens)

            # 每 3 个 batch 联合一次 2D 笔画图双线性边分类损失
            loss_total = loss_ctc
            if graph_train and step_idx % 3 == 0:
                g_samp = graph_train[int(graph_rng.integers(0, len(graph_train)))]
                sp_g = torch.from_numpy(g_samp.stroke_seqs).to(device)
                g_pmask = ~(sp_g.abs().sum(dim=-1) > 1e-6)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    _, sem_g, _ = model(sp_g, pad_mask=g_pmask)
                    idx_t = torch.from_numpy(g_samp.pair_indices).to(device)
                    geom_t = torch.from_numpy(g_samp.pair_geom).to(device)
                    lbl_t = torch.from_numpy(g_samp.pair_labels).to(device)
                    edge_logits = model.forward_edge_head(sem_g[idx_t[:, 0]], sem_g[idx_t[:, 1]], geom_t)
                loss_e = F.cross_entropy(edge_logits.float(), lbl_t)
                loss_total = loss_ctc + 0.35 * loss_e
                tr_edge_loss += float(loss_e.item())
                edge_steps += 1

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            tr_ctc_loss += float(loss_ctc.item())
        scheduler.step()

        real_val_cer, sample_preview = _eval_ctc_cer(model, va_loader, device, use_beam=False)
        edge_acc = _eval_edge_acc(model, graph_val, device)
        combined_score = real_val_cer - 0.15 * edge_acc
        if combined_score <= best_score:
            best_score = combined_score
            torch.save(model.state_dict(), out_path)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            tr_probe_cer, _ = _eval_ctc_cer(model, tr_probe_loader, device, use_beam=False)
            univ_cer, _ = _eval_ctc_cer(model, univ_va_loader, device, use_beam=False)
            print(
                f"[Stage 1] Epoch {epoch:02d}/{args.epochs} ({time.perf_counter()-ep_t0:.1f}s) | "
                f"ctc={tr_ctc_loss/max(1,len(tr_loader)):.4f} edge_ce={tr_edge_loss/max(1,edge_steps):.4f} | "
                f"train_CER={tr_probe_cer:.4f} real_val_CER={real_val_cer:.4f} univ_CER={univ_cer:.4f} | "
                f"edge_Acc={edge_acc:.2%} | {sample_preview}"
            )

    model.load_state_dict(torch.load(out_path, map_location=device), strict=True)
    final_edge_acc = _eval_edge_acc(model, graph_val, device)
    print(f"[Stage 1 完成] 最优权重={out_path} | Edge_Val_Acc={final_edge_acc:.2%}")
    if test_loader is not None:
        test_greedy_cer, _ = _eval_ctc_cer(model, test_loader, device, use_beam=False)
        test_beam_cer, preview_b = _eval_ctc_cer(model, test_loader, device, use_beam=True, beam_width=4, enforce_grammar=True)
        print(
            f"[Stage 1 留出测试集 (n={len(real_test_pairs)})] "
            f"Greedy_CER={test_greedy_cer:.4f} | TAMER-Grammar Beam4_CER={test_beam_cer:.4f} | {preview_b}"
        )

    if args.train_coverage:
        # 冻结编码器主干，微调 CoMER Coverage Attention Seq2Seq Decoder 供 ONNX 轨
        cov_dec = CoverageMathSeq2SeqDecoder(enc_dim=math_d_model, dec_dim=192, vocab_size=VOCAB_SIZE, dropout=0.1).to(device)
        cov_params = sum(p.numel() for p in cov_dec.parameters())
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        opt_cov = torch.optim.AdamW(cov_dec.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sos_id = VOCAB_SIZE
        eos_id = VOCAB_SIZE + 1
        cov_epochs = max(4, min(10, args.epochs // 3))
        print(f"[Stage 1b Coverage] 开始训练 CoverageMathSeq2SeqDecoder ({cov_params/1e6:.2f}M params, epochs={cov_epochs})...")
        for cep in range(1, cov_epochs + 1):
            cov_dec.train()
            tot_l = 0.0
            n_b = 0
            for x, pmask, _, tgts, tgt_lens, _ in tr_loader:
                x, pmask = x.to(device), pmask.to(device)
                with torch.no_grad():
                    z_sem, _, _ = model(x, pad_mask=pmask)
                bsz, sub_len, _ = z_sem.shape
                max_t = int(tgt_lens.max().item()) + 1
                seq_tgts = torch.full((bsz, max_t), eos_id, dtype=torch.long, device=device)
                off = 0
                for bi, tl in enumerate(tgt_lens.tolist()):
                    seq_tgts[bi, :tl] = tgts[off : off + tl].to(device)
                    off += tl

                h_dec = torch.zeros(2, bsz, cov_dec.dec_dim, device=device)
                cov_acc = torch.zeros(bsz, sub_len, device=device)
                prev_tok = torch.full((bsz,), sos_id, dtype=torch.long, device=device)
                step_losses = []
                for ti in range(max_t):
                    d_logits, h_dec, cov_acc = cov_dec.step(prev_tok, h_dec, z_sem, cov_acc)
                    step_losses.append(F.cross_entropy(d_logits, seq_tgts[:, ti]))
                    prev_tok = seq_tgts[:, ti]
                loss_c = torch.stack(step_losses).mean()
                opt_cov.zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(cov_dec.parameters(), 2.0)
                opt_cov.step()
                tot_l += float(loss_c.item())
                n_b += 1
            print(f"[Stage 1b Coverage] Epoch {cep:02d}/{cov_epochs} | ce_loss={tot_l/max(1,n_b):.4f}")
        cov_out = save_dir / "stage1_coverage_decoder.pt"
        torch.save(cov_dec.state_dict(), cov_out)
        print(f"[Stage 1b Coverage 完成] 权重已保存：{cov_out}")


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
        in_dim=DYNAMICS_DIM,
        d_model=int(MODEL_CONFIG_DEFAULTS["d_model"]),
        num_layers=int(MODEL_CONFIG_DEFAULTS["gru_layers"]),
        dropout=float(MODEL_CONFIG_DEFAULTS["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-4)

    out_path = save_dir / "stage2_causal_dynamics.pt"
    print(f"[Stage 2] 使用设备 {device} 预训练 {DYNAMICS_DIM} 维因果动力学+图谱状态自监督编码器...")
    for epoch in range(1, args.epochs + 1):
        enc.train()
        total_loss = 0.0
        for (dy,) in loader:
            dy = dy.to(device)
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
    import torch

    bsz, w_steps, l_s, c_in = stroke_sp_tensor.shape
    flat = stroke_sp_tensor.view(bsz * w_steps, l_s, c_in).to(device)
    non_empty_frames = flat.abs().sum(dim=-1) > 1e-6
    valid_stroke_mask = non_empty_frames.any(dim=-1)
    d_model = int(MODEL_CONFIG_DEFAULTS["math_d_model"])
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
        EdgeMathGraphNet,
        cognitive_cost_sensitive_loss,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    math_ckpt = Path(args.math_checkpoint or (save_dir / "stage1_edge_math_encoder.pt"))
    dyn_ckpt = Path(args.dyn_checkpoint or (save_dir / "stage2_causal_dynamics.pt"))

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
    ).to(device)
    if math_ckpt.exists():
        math_enc.load_state_dict(torch.load(math_ckpt, map_location=device), strict=True)
        print(f"[Stage 3] 加载已训练空间+图谱编码器：{math_ckpt}")
    math_enc.eval()
    for p in math_enc.parameters():
        p.requires_grad = False

    cog_net = CognitiveStateNet(MODEL_CONFIG_DEFAULTS).to(device)
    if dyn_ckpt.exists():
        cog_net.dynamics_encoder.load_state_dict(torch.load(dyn_ckpt, map_location=device), strict=True)
        print(f"[Stage 3] 加载已预训练 {DYNAMICS_DIM} 维因果动力学权重：{dyn_ckpt}")

    windows = collect_cognitive_windows(args.data_dir, n_syn_sessions=240)
    sp_steps = torch.tensor(np.stack([w.stroke_spatial_seqs for w in windows]), dtype=torch.float32)
    dy_steps = torch.tensor(np.stack([w.dynamics_seq for w in windows]), dtype=torch.float32)
    y_tensor = torch.tensor([w.label_id for w in windows], dtype=torch.long)
    mra_tensor = torch.tensor([[w.latent_m, w.latent_r, w.latent_a] for w in windows], dtype=torch.float32)

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

    print(f"[Stage 3] 使用设备 {device} 微调 {DYNAMICS_DIM} 维图谱融合 CognitiveStateNet (n_train={n_tr}, n_val={n_val})...")
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
    parser = argparse.ArgumentParser(description="《笔迹脉搏》PyTorch GPU 四阶段解耦 + 笔画关系图训练脚本")
    parser.add_argument(
        "--stage",
        choices=["spatial_pretrain", "temporal_ssl", "cognitive_finetune", "all"],
        default="all",
    )
    parser.add_argument("--mathwriting-path", type=str, default=None, help="已解压的 MathWriting 目录或 .tgz 路径")
    parser.add_argument("--mathwriting-tgz", type=str, default=None, help="MathWriting .tgz 数据路径（兼容旧参数）")
    parser.add_argument("--mw-max-samples", type=int, default=24000)
    parser.add_argument("--aug-samples", type=int, default=3000)
    parser.add_argument("--graph-samples", type=int, default=1800)
    parser.add_argument("--train-coverage", action="store_true", help="是否同时训练 CoverageMathSeq2SeqDecoder ONNX 可选解码器")
    parser.add_argument("--data-dir", type=str, default=None, help="App 导出的 v2.0-annotated JSON 目录")
    parser.add_argument("--save-dir", type=str, default="ml/runs")
    parser.add_argument("--math-checkpoint", type=str, default=None)
    parser.add_argument("--dyn-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=25)
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
