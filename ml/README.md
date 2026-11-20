# 《笔迹脉搏》端侧认知模型训练、数据管线与基准评测 (`ml/`)

本目录实现《笔迹脉搏》的**双流端侧模型训练、双轨部署导出与科学评测流水线**，特征提取、词表索引与推理计算与移动端 [src/engine/neuralRuntime.ts](../src/engine/neuralRuntime.ts) 逐维对齐。

## 目录结构

- `data_pipeline/vocab.py`：125 Token 大学数学 CTC 词表（高数/线代/概率论），提供贪心解码与 **CTC Prefix Beam Search（`beam=4`）**，与 [src/engine/mathVocab.ts](../src/engine/mathVocab.ts) 1-to-1 对齐。
- `data_pipeline/mathwriting_loader.py`：Google MathWriting InkML 解析器、**零截断自适应 6 通道等弧长重采样**（`max_len=192`）、大学数学增强公式生成器与多进程高速加载器。
- `data_pipeline/schema_converter.py`：解析 App 导出的 `v2.0-annotated` 会话 JSON（支持 `INVALID_INTERRUPT` 离屏干扰自动过滤），提取 **10 维无采样率依赖动力学脉冲序列**并对齐 6 类认知标签。
- `models/decoupled_dual_stream.py`：解耦双流网络：
  - `EdgeMathEncoder`：Linear + ConvSubsample(`stride=2`) + 无参正弦位置编码 + 4 层轻量 Conformer（`d=96`, 473.5K 参数）；
  - `CognitiveStateNet`：严格单向 2 层因果 GRU（`d=64`）+ `sem_proj(96→64)` 因果 Cross-Attention + 6 分类 / 惊奇度 / `[M, R, A]` 头，配合 `2.0×` 深度心算防误报惩罚损失。
- `train/train_pipeline.py`：三阶段 GPU 抗过拟合增强训练（Stage 1 空间 CTC 预训练 + SpecAugment/几何增强 → Stage 2 因果动力学自监督 → Stage 3 六分类代价敏感微调）。
- `train/evaluate_real_louo.py`：真实采集数据自动化入库质检、**按人隔离 LOUO-CV** 评测（`ThinkFPR / StuckF1 / PR-AUC / 6C_MacroF1 / Cohen's κ`）。
- `train/benchmark_baseline.py`：固定阈值 vs 多门控 `[M, R, A]` vs 双流融合模型的消融对比基准脚本。
- `export/export_onnx.py` & `export/export_ts_weights.py`：同步导出 ONNX（含 onnxruntime CPU 校验）与 Expo Go 纯 TS INT8 Base64 量化权重（[src/engine/edgeNeuralWeights.ts](../src/engine/edgeNeuralWeights.ts)）。

## 常用命令（WSL2 CUDA 环境）

```sh
# 1. 运行固定阈值 vs 状态机 vs 双流消融基准（纯 CPU 即可运行）
python ml/train/benchmark_baseline.py

# 2. 对真实标注目录执行入库质检与 LOUO 按人隔离交叉验证（若提供盲审目录则同步计算 Cohen's κ）
python ml/train/evaluate_real_louo.py --data-dir ml/real_data --blind-audit-dir ml/blind_audit

# 3. 使用真实标注数据微调 Stage 3 六分类认知网络（冻结 EdgeMathEncoder）
python ml/train/train_pipeline.py --stage cognitive_finetune --data-dir ml/real_data --epochs 35 --batch-size 32 --lr 1e-3 --save-dir ml/runs

# 4. 导出最新 ONNX 与移动端 INT8 量化权重
python ml/export/export_onnx.py --out-dir ml/runs/onnx
python ml/export/export_ts_weights.py
```

## 端侧双轨运行机制

- **默认轨（Expo Go 零原生依赖）**：应用开箱即用 Pure-TS `Float32Array` 核解包 `src/engine/edgeNeuralWeights.ts` 进行真机本地推理；
- **可选加速轨（自定义 Dev Client）**：可通过 `src/engine/types.ts` 预留的 `IInkSyntaxEncoderAdapter` 接入 `onnxruntime-react-native` 运行 `edge_math_encoder.onnx`。
