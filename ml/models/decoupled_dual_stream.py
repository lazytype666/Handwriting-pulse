"""
《笔迹脉搏》端侧轻量级解耦双流神经网络架构（PyTorch 确认版）。

包含两套既可独立运行、又可通过交叉注意力无缝联动的端侧网络（d_model=64，总参数量 < 260K，量化后 < 350KB）：
1. [模型 1] EdgeMathEncoder：
   - 输入 6 通道等弧长重采样空间序列 [dx, dy, cosθ, sinθ, pen_state, y_rel]（无时间）；
   - Stem Linear + ConvSubsample(stride=2, k=5) + 2 层轻量 Conformer Block；
   - 输出 CTC 大学数学 121 Token 识别对数几率 [B, L', 121] 与逐帧语义隐向量 Z_sem [B, L', 64]。
2. [模型 2] CognitiveStateNet：
   - 输入 10 维笔画决策步级非线性动力学流（含个人停顿 Z 分数、多尺度衰减积分 τ=1.5s/10s 等）；
   - 严格单向因果 GRU（2 层, d=64, 杜绝未来泄漏）+ Cross-Attention (Query=Z_dyn, Key/Value=Z_sem_step)；
   - 输出 6 类物理+多源可辨识状态 logits[6]、认知惊奇度 surprise[1] 与连续认知潜空间 [M, R, A]。
   - 训练损失包含“2.0× 深度心算防误报惩罚代价权重”。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

MODEL_CONFIG_DEFAULTS: Dict[str, int | float] = {
    "spatial_in_dim": 6,
    "dynamics_in_dim": 10,
    "math_d_model": 96,
    "d_model": 64,
    "n_heads": 4,
    "conformer_layers": 4,
    "gru_layers": 2,
    "num_classes": 6,
    "math_vocab_size": 125,
    "max_spatial_len": 192,
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

        def __init__(self, d_model: int = 64, n_heads: int = 4, kernel_size: int = 5, dropout: float = 0.1) -> None:
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


    class EdgeMathEncoder(nn.Module):
        """
        [模型 1] 大学数学全符号端侧识别与语义编码器（Spatial Conformer + CTC）。
        输入：6 通道无时间几何特征 [B, L, 6]，经 stride=2 卷积下采样至 [B, L//2, D]。
        """

        def __init__(
            self,
            in_dim: int = 6,
            d_model: int = 64,
            n_heads: int = 4,
            num_layers: int = 2,
            math_vocab_size: int = 121,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            # stride=2 下采样同时聚集局部笔段方向与降低 Attention O(L^2) 计算量 4 倍
            self.subsample = nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2)
            self.sub_norm = nn.LayerNorm(d_model)
            self.blocks = nn.ModuleList(
                [ConformerBlock(d_model, n_heads, kernel_size=5, dropout=dropout) for _ in range(num_layers)]
            )
            self.out_norm = nn.LayerNorm(d_model)
            self.ctc_head = nn.Linear(d_model, math_vocab_size)

        def forward(
            self,
            spatial_seq: torch.Tensor,
            pad_mask: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            返回:
              z_sem_seq: [B, L', D]
              sem_pooled: [B, D]
              ctc_logits: [B, L', V]
            """
            h = self.stem(spatial_seq).transpose(1, 2)  # [B, D, L]
            h = F.silu(self.subsample(h)).transpose(1, 2)  # [B, L', D]
            h = self.sub_norm(h)

            # 注入无参正弦位置编码（与移动端 Float32Array 内核逐位等价）
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
                # 防止全遮蔽行触发 MultiheadAttention Softmax NaN（纯布尔掩码避开 bool Where 算子以兼容 ONNX Runtime CPU）
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


    class CausalDynamicsEncoder(nn.Module):
        """
        [模型 2 上半部] 严格因果时序动力学编码器（只看历史 W 个决策步，严禁双向泄漏未来信息）。
        """

        def __init__(
            self,
            in_dim: int = 10,
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
            # 严格单向 GRU（bidirectional=False），保证离线训练与端侧逐帧流式推理零差异
            self.gru = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            # Stage 2 掩码脉冲自监督预测头（预测下一决策步动力学变化）
            self.ssl_next_head = nn.Linear(d_model, in_dim)

        def forward(self, dynamics_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.proj(dynamics_seq)
            z_dyn, _ = self.gru(h)
            next_pred = self.ssl_next_head(z_dyn)
            return z_dyn, next_pred


    class CognitiveStateNet(nn.Module):
        """
        [模型 2 完整版] 因果动力学流 + 空间语义步级交叉注意力 + 6 类认知状态/MRA 多任务网络。
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
            # 语义维度桥接投影（96 -> 64）：让公式识别器扩容的同时保持认知状态网络轻量化、防小样本过拟合
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
            """
            参数:
              dynamics_seq: [B, W, 10] （每个决策步的非线性能量/Z分数变换特征）
              sem_step_seq: [B, W, D_math] （由冻结的 EdgeMathEncoder 对对应笔画提取的语义向量序列）
            返回:
              logits: [B, 6]
              surprise: [B, 1]
              latent_mra: [B, 3] -> [M ∈ [0,1], R ∈ [0,1], A ∈ [-1,1]]
            """
            z_dyn, _ = self.dynamics_encoder(dynamics_seq)
            sem_k = self.sem_proj(sem_step_seq)
            # 构造严格因果注意力掩码：第 t 步动力学只能看第 <= t 步之前写下的符号，杜绝偷看未来！
            w_len = dynamics_seq.shape[1]
            causal_mask = torch.triu(
                torch.ones(w_len, sem_k.shape[1], device=dynamics_seq.device, dtype=torch.bool),
                diagonal=1,
            )
            fused, _ = self.cross_attn(
                z_dyn, sem_k, sem_k, attn_mask=causal_mask, need_weights=False
            )
            combined = torch.cat([z_dyn, fused], dim=-1)
            # 只取最后一个决策步的因果隐状态输出（对应实时判断时刻 t）
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
        - 对深度心算（class 1）错判成任何主动打扰类（class >= 2）施加 2.0x 额外罚项，死守防误报底线；
        - 联合拟合连续认知潜空间 [M, R, A] 与惊奇度 S(t)。
        """
        probs = F.softmax(logits, dim=-1)
        ce = F.cross_entropy(logits, targets)

        # 心算保护惩罚：若真实标签为 DEEP_THINKING_HOLD(1)，则惩罚模型给 class >= 2 分配概率之和
        think_mask = (targets == 1).float()
        stuck_prob_sum = probs[:, 2:].sum(dim=-1)
        think_fp_penalty = (think_mask * stuck_prob_sum).sum() / think_mask.sum().clamp(min=1.0)

        stuck_target = (targets >= 2).float().unsqueeze(-1)
        safe_surp = pred_surprise.clamp(1e-6, 1.0 - 1e-6)
        surp_loss = F.binary_cross_entropy(safe_surp, stuck_target)
        mra_loss = F.mse_loss(pred_mra, target_mra)

        return ce + 2.0 * think_fp_penalty + 0.35 * surp_loss + 0.35 * mra_loss
