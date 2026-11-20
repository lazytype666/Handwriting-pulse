# 《笔迹脉搏》端侧认知模型训练、数据管线与基准评测 (`ml/`)

本目录实现《笔迹脉搏》后续架构中的**离线表征学习与端侧部署管线**，与移动端 React Native / TypeScript 核心引擎（`src/engine/`）通过一致的特征格式（`v2.0-annotated` JSON 与 5 通道解耦空间流）完全对齐。

## 目录结构

- `data_pipeline/schema_converter.py`：解析移动端 App 导出的 `inkpulse-*.json` 会话（支持阶段一与 `v2.0-annotated`），提取 **5 维无时间空间几何流**与 **6 维个体归一化动力学流**，并内置快/慢手对比合成数据集。
- `models/decoupled_dual_stream.py`：PyTorch 解耦双流模型 `HandwritingPulseNet`（`SpatialConformerEncoder` + `TemporalDynamicsEncoder` + `CrossAttentionSurpriseFusion`）。
- `train/benchmark_baseline.py`：纯 NumPy / Scikit-Learn 本地可复现消融基准脚本，无需 GPU 即可对比“固定停顿阈值”、“个人 Z 分数状态机”和“双流融合分类器”的卡壳 F1 与**深度心算误报率**。
- `train/train_pipeline.py`：四阶段 PyTorch 训练管线（空间流 MathWriting 预训练 → 动力学流掩码自监督 → 100 题伴随标注小样本微调 → 评估）。
- `export/export_onnx.py`：动态尺寸 `.onnx` 导出与 `onnxruntime` 数值对齐/单步延时检验。

## 快速运行本地消融实验（无需 PyTorch/GPU）

在仓库根目录执行：

```sh
python ml/train/benchmark_baseline.py
```

若已从手机/平板端导出一批 `v2.0-annotated` 的真实演算 JSON 并放入 `./collected_sessions`，执行：

```sh
python ml/train/benchmark_baseline.py --data-dir ./collected_sessions
```

## PyTorch 完整四阶段训练与 ONNX 导出（GPU / Colab 环境）

```sh
pip install -r ml/requirements.txt

# 阶段 1：空间 Conformer 编码器几何预训练（接入 MathWriting 后接 CTC Loss）
python ml/train/train_pipeline.py --stage spatial_pretrain --epochs 10

# 阶段 2：动力学节奏编码器自监督掩码重建
python ml/train/train_pipeline.py --stage temporal_ssl --checkpoint ml/runs/stage1_spatial_conformer.pt --epochs 15

# 阶段 3：冻结浅层空间主干，基于伴随标注微调交叉注意力认知状态机
python ml/train/train_pipeline.py --stage cognitive_finetune --checkpoint ml/runs/stage2_temporal_ssl.pt --epochs 30

# 导出动态图 ONNX 并在 CPU 上校准精度与延时
python ml/export/export_onnx.py --checkpoint ml/runs/stage3_handwriting_pulse.pt --out-dir ml/runs/onnx
```

## 端侧 ONNX 模型接入手机端契约

移动端 `src/engine/types.ts` 已预留标准接口 `IInkSyntaxEncoderAdapter`，真机使用自定义 Dev Client 或端侧引擎时调用 `engine.registerSyntaxEncoderAdapter(customOnnxAdapter)` 即可无缝接管 `EdgeInkSyntaxEncoder` 的默认几何推理器；而在标准 Expo Go 环境下，应用会自动使用零第三方依赖的 `EdgeInkSyntaxEncoder` 完成实时符号/逻辑连接边界识别与步骤对齐。
