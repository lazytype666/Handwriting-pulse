"""
《笔迹脉搏》端侧轻量级解耦双流 + 笔画关系图神经网络架构（PyTorch 全面升级版）。

包含三大可导出 ONNX / INT8 的端侧模块：
1. [模型 1] EdgeMathGraphNet (= EdgeMathEncoder 增强版)：
   - 输入 10 通道无时间几何序列 [dx, dy, cosθ, sinθ, pen_state, y_rel, st_cx_rel, st_cy_rel, st_w_norm, st_h_norm]；
   - Stem Linear + ConvSubsample(stride=2, k=5) + 4 层轻量 Conformer Block (d=96)；
   - 输出 125 词表 CTC 对数几率、逐帧语义特征 Z_sem [B, L', 96]；
   - 新增双线性笔画图边分类头 `StrokeGraphEdgeHead`（仅 ~17.1K 参数），联合 10 维成对几何差值预测 7 类二维边关系
     (NONE / SAME_SYMBOL / RIGHT / ABOVE / BELOW / SUP / SUB)，彻底打破纯手工聚类的阈值脆性。
2. [模型 2] CognitiveStateNet：
   - 输入 14 维因果动力学与解题图谱联合特征流（前 10 维个人节奏/涂改积分 + 后 4 维 [anchorGap, stepStagnation, stepAdvancePulse, stepProgress]）；
   - 严格单向因果 GRU（2 层, d=64）+ Cross-Attention (Query=Z_dyn, Key/Value=Z_sem_step)；
   - 输出 6 类认知状态 logits[6]、惊奇度 surprise[1] 与连续认知潜空间 [M, R, A]。
3. [可选加速轨] CoverageMathSeq2SeqDecoder：
   - 借鉴 CoMER 的带位置覆盖（Coverage Attention）轻量 2 层 GRU 公式树序列化解码器（约 ~2.1M 参数），
     供 Custom Dev Client / APK 通过 ONNX Runtime 原生算子按需启用。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

MODEL_CONFIG_DEFAULTS: Dict[str, int | float] = {
    "spatial_in_dim": 10,
    "dynamics_in_dim": 14,
    "math_d_model": 96,
    "d_model": 64,
    "n_heads": 4,
    "conformer_layers": 4,
    "gru_layers": 2,
    "num_classes": 6,
    "math_vocab_size": 125,
    "max_spatial_len": 192,
    "edge_geom_dim": 10,
    "edge_hidden_dim": 48,
    "edge_classes": 7,
    "dropout": 0.12,
}

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ConformerBlock(nn.Module):
        """端侧轻量 Conformer Block（Multi-Head Self-Attention + Depthwise Conv1d + Swish FFN）。"""

        def __init__(self, d_model: int = 96, n_heads: int = 4, kernel_size: int = 5, dropout: float = 0.1) -> None:
            super().__init__()
            self.norm_attn = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm_conv = nn.LayerNorm(d_model)
            self.pw_in = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
            self.dw_conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=d_model,
            )
            self.pw_out = nn.Conv1d(d_model, d_model, kernel_size=1)
            self.norm_ffn = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
            h = self.norm_attn(x)
            attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + self.drop(attn_out)

            c = self.norm_conv(x).transpose(1, 2)
            a, b = self.pw_in(c).chunk(2, dim=1)
            c = a * torch.sigmoid(b)
            c = F.silu(self.dw_conv(c))
            c = self.pw_out(c).transpose(1, 2)
            x = x + self.drop(c)

            x = x + 0.5 * self.drop(self.ffn(self.norm_ffn(x)))
            return x


    class StrokeGraphEdgeHead(nn.Module):
        """
        极轻量双线性笔画图边关系分类头（~17.1K 参数）。
        输入两笔的语义嵌入 (sem_i, sem_j ∈ R^96) 与归一化成对几何特征 geom_ij ∈ R^10，
        预测 7 类二维关系：NONE / SAME_SYMBOL / RIGHT / ABOVE / BELOW / SUP / SUB。
        """

        def __init__(
            self,
            sem_dim: int = 96,
            geom_dim: int = 10,
            hidden_dim: int = 48,
            num_classes: int = 7,
        ) -> None:
            super().__init__()
            self.proj_u = nn.Linear(sem_dim, hidden_dim)
            self.proj_v = nn.Linear(sem_dim, hidden_dim)
            self.mlp_0 = nn.Linear(hidden_dim * 3 + geom_dim, hidden_dim)
            self.mlp_1 = nn.Linear(hidden_dim, num_classes)

        def forward(
            self,
            sem_i: torch.Tensor,
            sem_j: torch.Tensor,
            geom_ij: torch.Tensor,
        ) -> torch.Tensor:
            u = self.proj_u(sem_i)
            v = self.proj_v(sem_j)
            uv = u * v
            feat = torch.cat([u, v, uv, geom_ij], dim=-1)
            h = F.silu(self.mlp_0(feat))
            return self.mlp_1(h)


    class EdgeMathEncoder(nn.Module):
        """
        [模型 1 — EdgeMathGraphNet] 大学数学全符号端侧识别 + 二维笔画图关系编码器（10 通道 Conformer + CTC + EdgeHead）。
        输入：10 通道无时间几何特征 [B, L, 10]，经 stride=2 卷积下采样至 [B, L//2, D]。
        """

        def __init__(
            self,
            in_dim: int = 10,
            d_model: int = 96,
            n_heads: int = 4,
            num_layers: int = 4,
            math_vocab_size: int = 125,
            edge_geom_dim: int = 10,
            edge_hidden_dim: int = 48,
            edge_classes: int = 7,
            dropout: float = 0.12,
        ) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            self.subsample = nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2)
            self.sub_norm = nn.LayerNorm(d_model)
            self.blocks = nn.ModuleList(
                [ConformerBlock(d_model, n_heads, kernel_size=5, dropout=dropout) for _ in range(num_layers)]
            )
            self.out_norm = nn.LayerNorm(d_model)
            self.ctc_head = nn.Linear(d_model, math_vocab_size)
            self.edge_head = StrokeGraphEdgeHead(
                sem_dim=d_model,
                geom_dim=edge_geom_dim,
                hidden_dim=edge_hidden_dim,
                num_classes=edge_classes,
            )

        def forward(
            self,
            spatial_seq: torch.Tensor,
            pad_mask: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.stem(spatial_seq).transpose(1, 2)
            h = F.silu(self.subsample(h)).transpose(1, 2)
            h = self.sub_norm(h)

            seq_len = h.shape[1]
            half_dim = h.shape[2] // 2
            pos = torch.arange(seq_len, device=h.device, dtype=h.dtype).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, half_dim, device=h.device, dtype=h.dtype) * (-math.log(10000.0) / max(1, half_dim))
            )
            angles = pos * div.unsqueeze(0)
            pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            h = h + pe.unsqueeze(0)

            sub_mask = pad_mask[:, ::2] if pad_mask is not None else None
            if sub_mask is not None:
                if sub_mask.shape[1] != h.shape[1]:
                    sub_mask = sub_mask[:, : h.shape[1]]
                all_masked = sub_mask.all(dim=1, keepdim=True)
                pos_idx = torch.arange(sub_mask.shape[1], device=sub_mask.device).unsqueeze(0)
                clear_first = (~all_masked) | (pos_idx > 0)
                sub_mask = sub_mask & clear_first

            for block in self.blocks:
                h = block(h, key_padding_mask=sub_mask)

            z_sem_seq = self.out_norm(h)
            ctc_logits = self.ctc_head(z_sem_seq)

            if sub_mask is not None:
                valid = (~sub_mask).unsqueeze(-1).float()
                sem_pooled = (z_sem_seq * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            else:
                sem_pooled = z_sem_seq.mean(dim=1)
            return z_sem_seq, sem_pooled, ctc_logits

        def forward_edge_head(
            self,
            sem_i: torch.Tensor,
            sem_j: torch.Tensor,
            geom_ij: torch.Tensor,
        ) -> torch.Tensor:
            return self.edge_head(sem_i, sem_j, geom_ij)


    EdgeMathGraphNet = EdgeMathEncoder


    class CoverageMathSeq2SeqDecoder(nn.Module):
        """
        [可选加速轨 — Quote ④] 基于 Coverage Attention 的轻量 Seq2Seq 公式解码器（~2.1M 参数）。
        复用 Conformer 编码序列 `z_sem_seq`，通过卷积覆盖记忆消除长公式漏字/重复，用于 ONNX Runtime 原生加速轨。
        """

        def __init__(
            self,
            enc_dim: int = 96,
            dec_dim: int = 192,
            vocab_size: int = 125,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.dec_dim = dec_dim
            self.embed = nn.Embedding(vocab_size + 2, dec_dim)  # +SOS, +EOS
            self.enc_proj = nn.Linear(enc_dim, dec_dim)
            self.attn_q = nn.Linear(dec_dim, dec_dim, bias=False)
            self.cov_conv = nn.Conv1d(1, 32, kernel_size=7, padding=3)
            self.attn_cov = nn.Linear(32, dec_dim, bias=False)
            self.attn_v = nn.Linear(dec_dim, 1, bias=False)
            self.gru = nn.GRU(dec_dim + enc_dim, dec_dim, num_layers=2, batch_first=True, dropout=dropout)
            self.out_proj = nn.Linear(dec_dim + enc_dim + dec_dim, vocab_size + 2)

        def step(
            self,
            prev_token_id: torch.Tensor,  # [B]
            prev_hidden: torch.Tensor,    # [2, B, dec_dim]
            z_sem_seq: torch.Tensor,      # [B, L', enc_dim]
            cov_accum: torch.Tensor,      # [B, L']
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            emb = self.embed(prev_token_id)  # [B, dec_dim]
            enc_k = self.enc_proj(z_sem_seq)  # [B, L', dec_dim]
            cov_feat = self.cov_conv(cov_accum.unsqueeze(1)).transpose(1, 2)  # [B, L', 32]
            q = self.attn_q(prev_hidden[-1]).unsqueeze(1)
            energy = torch.tanh(enc_k + q + self.attn_cov(cov_feat))
            scores = self.attn_v(energy).squeeze(-1)  # [B, L']
            alpha = F.softmax(scores, dim=-1)
            ctx = torch.bmm(alpha.unsqueeze(1), z_sem_seq).squeeze(1)  # [B, enc_dim]
            next_cov = cov_accum + alpha

            gru_in = torch.cat([emb, ctx], dim=-1).unsqueeze(1)
            gru_out, next_hidden = self.gru(gru_in, prev_hidden)
            h_dec = gru_out.squeeze(1)
            logits = self.out_proj(torch.cat([h_dec, ctx, emb], dim=-1))
            return logits, next_hidden, next_cov

        def forward(
            self,
            prev_token_id: torch.Tensor,
            prev_hidden: torch.Tensor,
            z_sem_seq: torch.Tensor,
            cov_accum: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.step(prev_token_id, prev_hidden, z_sem_seq, cov_accum)


    class CausalDynamicsEncoder(nn.Module):
        """
        [模型 2 上半部] 严格因果时序动力学编码器（只看历史 W 个决策步，严禁双向泄漏未来信息）。
        """

        def __init__(
            self,
            in_dim: int = 14,
            d_model: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            self.gru = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.ssl_next_head = nn.Linear(d_model, in_dim)

        def forward(self, dynamics_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.proj(dynamics_seq)
            z_dyn, _ = self.gru(h)
            next_pred = self.ssl_next_head(z_dyn)
            return z_dyn, next_pred


    class CognitiveStateNet(nn.Module):
        """
        [模型 2 完整版] 14 维因果动力学与解题图谱联合流 + 空间语义步级交叉注意力 + 6 类认知状态/MRA 多任务网络。
        """

        def __init__(self, cfg: Dict[str, int | float] | None = None) -> None:
            super().__init__()
            c = {**MODEL_CONFIG_DEFAULTS, **(cfg or {})}
            d_model = int(c["d_model"])
            math_d_model = int(c.get("math_d_model", d_model))
            n_heads = int(c["n_heads"])
            self.dynamics_encoder = CausalDynamicsEncoder(
                in_dim=int(c["dynamics_in_dim"]),
                d_model=d_model,
                num_layers=int(c["gru_layers"]),
                dropout=float(c["dropout"]),
            )
            self.sem_proj = nn.Linear(math_d_model, d_model)
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=float(c["dropout"]), batch_first=True
            )
            self.fuse_norm = nn.LayerNorm(d_model * 2)
            self.mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.SiLU(),
                nn.Dropout(float(c["dropout"])),
            )
            self.class_head = nn.Linear(d_model, int(c["num_classes"]))
            self.surprise_head = nn.Linear(d_model, 1)
            self.mra_head = nn.Linear(d_model, 3)

        def forward(
            self,
            dynamics_seq: torch.Tensor,
            sem_step_seq: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            z_dyn, _ = self.dynamics_encoder(dynamics_seq)
            sem_k = self.sem_proj(sem_step_seq)
            w_len = dynamics_seq.shape[1]
            causal_mask = torch.triu(
                torch.ones(w_len, sem_k.shape[1], device=dynamics_seq.device, dtype=torch.bool),
                diagonal=1,
            )
            fused, _ = self.cross_attn(
                z_dyn, sem_k, sem_k, attn_mask=causal_mask, need_weights=False
            )
            combined = torch.cat([z_dyn, fused], dim=-1)
            latest = self.fuse_norm(combined[:, -1, :])
            feat = self.mlp(latest)
            logits = self.class_head(feat)
            surprise = torch.sigmoid(self.surprise_head(feat))
            raw_mra = self.mra_head(feat)
            m_val = torch.sigmoid(raw_mra[:, 0:1])
            r_val = torch.sigmoid(raw_mra[:, 1:2])
            a_val = torch.tanh(raw_mra[:, 2:3])
            latent_mra = torch.cat([m_val, r_val, a_val], dim=-1)
            return logits, surprise, latent_mra


    def cognitive_cost_sensitive_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        pred_surprise: torch.Tensor,
        pred_mra: torch.Tensor,
        target_mra: torch.Tensor,
    ) -> torch.Tensor:
        """
        多任务代价敏感损失函数：
        - 对深度心算（class 1）错判成任何主动打扰类（class >= 2）施加 2.0x 额外罚项；
        - 联合拟合连续认知潜空间 [M, R, A] 与惊奇度 S(t)。
        """
        probs = F.softmax(logits, dim=-1)
        ce = F.cross_entropy(logits, targets)

        think_mask = (targets == 1).float()
        stuck_prob_sum = probs[:, 2:].sum(dim=-1)
        think_fp_penalty = (think_mask * stuck_prob_sum).sum() / think_mask.sum().clamp(min=1.0)

        stuck_target = (targets >= 2).float().unsqueeze(-1)
        safe_surp = pred_surprise.clamp(1e-6, 1.0 - 1e-6)
        surp_loss = F.binary_cross_entropy(safe_surp, stuck_target)
        mra_loss = F.mse_loss(pred_mra, target_mra)

        return ce + 2.0 * think_fp_penalty + 0.35 * surp_loss + 0.35 * mra_loss
