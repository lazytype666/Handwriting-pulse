"""
《笔迹脉搏》全模型 ONNX 导出与 onnxruntime 一致性基准测试脚本（10 通道图编码 + 笔画边关系 + Coverage 解码 + 14 维因果状态机）。

导出产物（位于 --out-dir）：
1. `edge_math_encoder.onnx`           （10 通道 Conformer-CTC 识别与单笔语义编码器）
2. `edge_math_graph_edges.onnx`       （双线性笔画图 7 类关系边分类头）
3. `edge_math_coverage_decoder.onnx`  （CoMER 风格 Coverage Attention Seq2Seq 单步解码器）
4. `cognitive_state_net.onnx`         （14 维因果动力学+图谱融合 6 分类 / [M, R, A] / 惊奇度网络）
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data_pipeline.vocab import VOCAB_SIZE  # noqa: E402
from ml.models.decoupled_dual_stream import MODEL_CONFIG_DEFAULTS, TORCH_AVAILABLE  # noqa: E402


def _export_with_compat(model, args_tuple, out_path: Path, input_names, output_names, dynamic_axes) -> None:
    import torch

    sig = inspect.signature(torch.onnx.export)
    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
    if "dynamo" in sig.parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(model, args_tuple, str(out_path), **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="《笔迹脉搏》ONNX 导出与 ORT 一致性检验")
    parser.add_argument("--math-ckpt", type=str, default="ml/runs/stage1_edge_math_encoder.pt")
    parser.add_argument("--cov-ckpt", type=str, default="ml/runs/stage1_coverage_decoder.pt")
    parser.add_argument("--cog-ckpt", type=str, default="ml/runs/stage3_cognitive_state_net.pt")
    parser.add_argument("--out-dir", type=str, default="ml/runs/onnx")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit("当前 Python 环境未安装 PyTorch。请在 WSL2 环境中使用 /root/.inkpulse_venv/bin/python 运行。")

    import torch
    from ml.models.decoupled_dual_stream import (
        CognitiveStateNet,
        CoverageMathSeq2SeqDecoder,
        EdgeMathGraphNet,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spatial_dim = int(MODEL_CONFIG_DEFAULTS["spatial_in_dim"])
    dyn_dim = int(MODEL_CONFIG_DEFAULTS["dynamics_in_dim"])
    math_d_model = int(MODEL_CONFIG_DEFAULTS["math_d_model"])

    # 1. 导出 EdgeMathGraphNet 主干
    math_enc = EdgeMathGraphNet(
        in_dim=spatial_dim,
        d_model=math_d_model,
        n_heads=int(MODEL_CONFIG_DEFAULTS["n_heads"]),
        num_layers=int(MODEL_CONFIG_DEFAULTS["conformer_layers"]),
        math_vocab_size=VOCAB_SIZE,
        edge_geom_dim=int(MODEL_CONFIG_DEFAULTS["edge_geom_dim"]),
        edge_hidden_dim=int(MODEL_CONFIG_DEFAULTS["edge_hidden_dim"]),
        edge_classes=int(MODEL_CONFIG_DEFAULTS["edge_classes"]),
        dropout=0.0,
    )
    if Path(args.math_ckpt).exists():
        math_enc.load_state_dict(torch.load(args.math_ckpt, map_location="cpu"), strict=True)
        print(f"[加载] EdgeMathGraphNet: {args.math_ckpt}")
    math_enc.eval()

    dummy_sp = torch.randn(1, 64, spatial_dim)
    dummy_mask = torch.zeros(1, 64, dtype=torch.bool)
    math_onnx = out_dir / "edge_math_encoder.onnx"
    _export_with_compat(
        math_enc,
        (dummy_sp, dummy_mask),
        math_onnx,
        ["spatial_seq", "pad_mask"],
        ["z_sem_seq", "sem_pooled", "ctc_logits"],
        {
            "spatial_seq": {0: "batch", 1: "seq_len"},
            "pad_mask": {0: "batch", 1: "seq_len"},
            "z_sem_seq": {0: "batch", 1: "sub_len"},
            "sem_pooled": {0: "batch"},
            "ctc_logits": {0: "batch", 1: "sub_len"},
        },
    )

    # 2. 导出 StrokeGraphEdgeHead
    edge_onnx = out_dir / "edge_math_graph_edges.onnx"
    dummy_si = torch.randn(8, math_d_model)
    dummy_sj = torch.randn(8, math_d_model)
    dummy_gm = torch.randn(8, int(MODEL_CONFIG_DEFAULTS["edge_geom_dim"]))
    _export_with_compat(
        math_enc.edge_head,
        (dummy_si, dummy_sj, dummy_gm),
        edge_onnx,
        ["sem_i", "sem_j", "geom_ij"],
        ["edge_logits"],
        {
            "sem_i": {0: "num_pairs"},
            "sem_j": {0: "num_pairs"},
            "geom_ij": {0: "num_pairs"},
            "edge_logits": {0: "num_pairs"},
        },
    )

    # 3. 导出 CoverageMathSeq2SeqDecoder
    cov_dec = CoverageMathSeq2SeqDecoder(enc_dim=math_d_model, dec_dim=192, vocab_size=VOCAB_SIZE, dropout=0.0)
    if Path(args.cov_ckpt).exists():
        cov_dec.load_state_dict(torch.load(args.cov_ckpt, map_location="cpu"), strict=True)
        print(f"[加载] CoverageMathSeq2SeqDecoder: {args.cov_ckpt}")
    cov_dec.eval()
    cov_onnx = out_dir / "edge_math_coverage_decoder.onnx"
    dummy_tok = torch.tensor([VOCAB_SIZE], dtype=torch.long)
    dummy_hid = torch.zeros(2, 1, 192)
    dummy_z = torch.randn(1, 32, math_d_model)
    dummy_cov = torch.zeros(1, 32)
    _export_with_compat(
        cov_dec,
        (dummy_tok, dummy_hid, dummy_z, dummy_cov),
        cov_onnx,
        ["prev_token_id", "prev_hidden", "z_sem_seq", "cov_accum"],
        ["step_logits", "next_hidden", "next_cov"],
        {
            "prev_token_id": {0: "batch"},
            "prev_hidden": {1: "batch"},
            "z_sem_seq": {0: "batch", 1: "sub_len"},
            "cov_accum": {0: "batch", 1: "sub_len"},
            "step_logits": {0: "batch"},
            "next_hidden": {1: "batch"},
            "next_cov": {0: "batch", 1: "sub_len"},
        },
    )

    # 4. 导出 CognitiveStateNet
    cog_net = CognitiveStateNet(MODEL_CONFIG_DEFAULTS)
    if Path(args.cog_ckpt).exists():
        cog_net.load_state_dict(torch.load(args.cog_ckpt, map_location="cpu"), strict=True)
        print(f"[加载] CognitiveStateNet: {args.cog_ckpt}")
    cog_net.eval()

    dummy_dyn = torch.randn(1, 20, dyn_dim)
    dummy_sem = torch.randn(1, 20, math_d_model)
    cog_onnx = out_dir / "cognitive_state_net.onnx"
    _export_with_compat(
        cog_net,
        (dummy_dyn, dummy_sem),
        cog_onnx,
        ["dynamics_seq", "sem_step_seq"],
        ["logits", "surprise", "latent_mra"],
        {
            "dynamics_seq": {0: "batch", 1: "win_steps"},
            "sem_step_seq": {0: "batch", 1: "win_steps"},
            "logits": {0: "batch"},
            "surprise": {0: "batch"},
            "latent_mra": {0: "batch"},
        },
    )

    print(
        f"[导出完成] math={math_onnx.stat().st_size/1024:.1f}KB | edge={edge_onnx.stat().st_size/1024:.1f}KB | "
        f"cov_dec={cov_onnx.stat().st_size/1024:.1f}KB | cog={cog_onnx.stat().st_size/1024:.1f}KB"
    )

    try:
        import onnxruntime as ort

        sess_m = ort.InferenceSession(str(math_onnx), providers=["CPUExecutionProvider"])
        sess_e = ort.InferenceSession(str(edge_onnx), providers=["CPUExecutionProvider"])
        sess_cov = ort.InferenceSession(str(cov_onnx), providers=["CPUExecutionProvider"])
        sess_c = ort.InferenceSession(str(cog_onnx), providers=["CPUExecutionProvider"])

        with torch.no_grad():
            _, _, pt_ctc = math_enc(dummy_sp, dummy_mask)
            pt_edge = math_enc.forward_edge_head(dummy_si, dummy_sj, dummy_gm)
            pt_dl, _, _ = cov_dec(dummy_tok, dummy_hid, dummy_z, dummy_cov)
            pt_logits, _, _ = cog_net(dummy_dyn, dummy_sem)

        t0 = time.perf_counter()
        _, _, ort_ctc = sess_m.run(None, {"spatial_seq": dummy_sp.numpy(), "pad_mask": dummy_mask.numpy()})
        t1 = time.perf_counter()
        ort_edge = sess_e.run(None, {"sem_i": dummy_si.numpy(), "sem_j": dummy_sj.numpy(), "geom_ij": dummy_gm.numpy()})[0]
        t2 = time.perf_counter()
        ort_dl, _, _ = sess_cov.run(
            None,
            {
                "prev_token_id": dummy_tok.numpy(),
                "prev_hidden": dummy_hid.numpy(),
                "z_sem_seq": dummy_z.numpy(),
                "cov_accum": dummy_cov.numpy(),
            },
        )
        t3 = time.perf_counter()
        ort_logits, _, _ = sess_c.run(None, {"dynamics_seq": dummy_dyn.numpy(), "sem_step_seq": dummy_sem.numpy()})
        t4 = time.perf_counter()

        print(
            f"[ORT 对齐校验] Math={float(np.max(np.abs(pt_ctc.numpy()-ort_ctc))):.2e} ({(t1-t0)*1000:.2f}ms) | "
            f"Edge={float(np.max(np.abs(pt_edge.numpy()-ort_edge))):.2e} ({(t2-t1)*1000:.2f}ms) | "
            f"CovDec={float(np.max(np.abs(pt_dl.numpy()-ort_dl))):.2e} ({(t3-t2)*1000:.2f}ms) | "
            f"Cog={float(np.max(np.abs(pt_logits.numpy()-ort_logits))):.2e} ({(t4-t3)*1000:.2f}ms)"
        )
    except ImportError:
        print("[提示] 未安装 onnxruntime，跳过本地推理校验。")


if __name__ == "__main__":
    main()
