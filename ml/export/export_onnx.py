"""
《笔迹脉搏》双模型 ONNX 导出、INT8 动态量化与 onnxruntime 一致性基准测试脚本。

导出产物（位于 --out-dir）：
1. `edge_math_encoder.onnx`      （大学数学 121 Token CTC 识别与语义编码器）
2. `cognitive_state_net.onnx`    （严格因果 6 分类认知状态 + [M, R, A] + 惊奇度网络）
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
    parser = argparse.ArgumentParser(description="《笔迹脉搏》双模型 ONNX 导出与推理检验")
    parser.add_argument("--math-ckpt", type=str, default="ml/runs/stage1_edge_math_encoder.pt")
    parser.add_argument("--cog-ckpt", type=str, default="ml/runs/stage3_cognitive_state_net.pt")
    parser.add_argument("--out-dir", type=str, default="ml/runs/onnx")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit("当前 Python 环境未安装 PyTorch。请在 WSL2 环境中使用 /root/.inkpulse_venv/bin/python 运行。")

    import torch
    from ml.models.decoupled_dual_stream import CognitiveStateNet, EdgeMathEncoder

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 导出 EdgeMathEncoder
    math_d_model = int(MODEL_CONFIG_DEFAULTS.get("math_d_model", MODEL_CONFIG_DEFAULTS["d_model"]))
    math_enc = EdgeMathEncoder(
        in_dim=int(MODEL_CONFIG_DEFAULTS["spatial_in_dim"]),
        d_model=math_d_model,
        n_heads=int(MODEL_CONFIG_DEFAULTS["n_heads"]),
        num_layers=int(MODEL_CONFIG_DEFAULTS["conformer_layers"]),
        math_vocab_size=VOCAB_SIZE,
        dropout=0.0,
    )
    if Path(args.math_ckpt).exists():
        math_enc.load_state_dict(torch.load(args.math_ckpt, map_location="cpu"), strict=True)
        print(f"[加载] EdgeMathEncoder: {args.math_ckpt}")
    math_enc.eval()

    dummy_sp = torch.randn(1, 64, 6)
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

    # 2. 导出 CognitiveStateNet
    cog_net = CognitiveStateNet(MODEL_CONFIG_DEFAULTS)
    if Path(args.cog_ckpt).exists():
        cog_net.load_state_dict(torch.load(args.cog_ckpt, map_location="cpu"), strict=True)
        print(f"[加载] CognitiveStateNet: {args.cog_ckpt}")
    cog_net.eval()

    dummy_dyn = torch.randn(1, 20, 10)
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

    math_kb = math_onnx.stat().st_size / 1024.0
    cog_kb = cog_onnx.stat().st_size / 1024.0
    print(f"[导出完成] edge_math_encoder.onnx ({math_kb:.1f} KB), cognitive_state_net.onnx ({cog_kb:.1f} KB)")

    try:
        import onnxruntime as ort

        sess_m = ort.InferenceSession(str(math_onnx), providers=["CPUExecutionProvider"])
        sess_c = ort.InferenceSession(str(cog_onnx), providers=["CPUExecutionProvider"])

        with torch.no_grad():
            pt_z, pt_pool, pt_ctc = math_enc(dummy_sp, dummy_mask)
            pt_logits, pt_surp, pt_mra = cog_net(dummy_dyn, dummy_sem)

        t0 = time.perf_counter()
        ort_z, ort_pool, ort_ctc = sess_m.run(
            None, {"spatial_seq": dummy_sp.numpy(), "pad_mask": dummy_mask.numpy()}
        )
        t1 = time.perf_counter()
        ort_logits, ort_surp, ort_mra = sess_c.run(
            None, {"dynamics_seq": dummy_dyn.numpy(), "sem_step_seq": dummy_sem.numpy()}
        )
        t2 = time.perf_counter()

        diff_m = float(np.max(np.abs(pt_ctc.numpy() - ort_ctc)))
        diff_c = float(np.max(np.abs(pt_logits.numpy() - ort_logits)))
        print(
            f"[ORT 对齐校验] MathEncoder max_diff={diff_m:.2e} ({(t1-t0)*1000:.2f} ms) | "
            f"CognitiveNet max_diff={diff_c:.2e} ({(t2-t1)*1000:.2f} ms)"
        )
    except ImportError:
        print("[提示] 未安装 onnxruntime，跳过本地推理校验。")


if __name__ == "__main__":
    main()
