# 笔迹脉搏 · Handwriting Pulse

笔迹脉搏是一张面向大学数学（高等数学、线性代数、概率统计）的安卓智能草稿纸。学生可粘贴题目或拍照/选图调用商用多模态大模型识别题目并自动预解**动态解题图谱与易错点检测 Logits**；在手写演算过程中，端侧双流神经网络与 14 维因果状态机联合评估书写动力学与图谱步进对齐度，主动保护「深度心算」并针对真正卡壳或歧路触发温和引导；单题完成后，端侧个性化学习智能体自动更新过程性知识追踪（Process-BKT）与书写习惯数字孪生。

> 配套开发与架构文档：[前后端完整开发需求文档 (Dev PRD)](docs/FULLSTACK_DEV_REQUIREMENTS.md) · [个性化学习智能体技术说明书](docs/LEARNING_AGENT_SPEC.md)

## 在安卓真机上运行

电脑准备 Node.js 24.x（至少 24.3.0）与 npm，安卓手机或平板安装与 Expo SDK 58 匹配的 Expo Go。设备连接同一个可互访的 Wi-Fi。从仓库根目录启动本地开发服务，用 Expo Go 扫描终端二维码。[下载兼容的 Expo Go](https://expo.dev/go?sdkVersion=58&platform=android&device=true)

```sh
git clone https://github.com/lazytype666/Handwriting-pulse.git
cd Handwriting-pulse
npm install
npm run typecheck
npm start
```

初次运行无需任何密钥即可体验内置 6 道高数/线代/概率离线解题图谱示例、10 通道端侧手写识别/边图推理与 14 维因果状态机。点「题干」可粘贴自定义题目或选择离线示例，留空也可直接书写、擦除、撤销、重做、清空、查看历史与导出。

## 拍照与相册识别题目（VLM 优先 + 一键原图重识）

「题干」编辑器提供「拍照识题」和「相册识题」，裁剪选区后将最长边压缩至 `1280px` JPEG 执行识别：

1. **商用 VLM 多模态识题（推荐，Expo Go 直接可用）**：配置统一商用 API 后（见下一节，独立指定 `EXPO_PUBLIC_VLM_MODEL` 或直接复用多模态 `EXPO_PUBLIC_LLM_*`），系统直走 `vlm_only` 高精度公式识题（超时 `15s` + 1 次瞬态重试，不再向 VLM 混入粗糙规则文本干扰判断）。若因网络抖动失败，界面保留原裁剪图并支持一键**「用原裁剪图重识」**，无需重新拍照裁剪。
2. **本机 ML Kit 降级兜底**：未配置云端 VLM 且运行在包含原生依赖的 Dev Client / 独立 APK 中时，才会调用设备端内置中文 ML Kit OCR 识别印刷体文本（Expo Go 无原生 ML Kit 模块）。

识别结果会填入预览区，核对公式、上下标与符号后点「完成」，系统会在后台静默为该题生成专属动态解题图谱与 Logits 检验头。

## 从书写到提示

1. 用「笔」写算式。当触发「停顿涂改」「速度骤降」或 14 维神经状态机认知阻塞门限（如结构边界长停、步骤长期滞后 `graphStepStagnationStall` 或高概率易错歧路）时，底部会弹出分层引导卡片（微卡 `L0` 微轻推 → `L1` 方向点拨 → `L2` 分步提示 → `L3` 主干路线）。
2. 若在正常数字心算中间短暂停顿，或刚刚跨过关键步骤后的脉冲保护窗口（`stepAdvancePulse`），状态机主动压低惊奇度并抑制弹窗误报。
3. 打开「历史」可回看本次会话的触发事件、微标注统计与个性化策略权重。
4. 「撤销」按笔回退，「重做」按原顺序恢复被撤销的笔画；撤销后重新落笔，重做栈清空；点「清空」并确认后清空画布并重置当前题追踪状态，已发生的卡壳历史仍保留。
5. 「采样诊断」展示近 1 秒真实触点率、点间隔及停顿/涂改/速度阈值滑块。

## 配置商用大模型 API 与智能体后端

将仓库根目录的 [.env.example](.env.example) 复制为 `.env.local`，按需填写后重启 Metro 并重载手机端（留空即使用全离线基线）：

| 环境变量 | 填写内容 | 作用范围 |
| --- | --- | --- |
| `EXPO_PUBLIC_LLM_BASE_URL` | 服务商 OpenAI 兼容基础地址（如 DeepSeek / 通义千问 / Gemini 兼容网关） | 商用大模型解题图谱预解、直连教练报告 |
| `EXPO_PUBLIC_LLM_API_KEY` | 演示用密钥（仅限本地或受控测试构建，生产建议走后端代理） | 商业接口鉴权 |
| `EXPO_PUBLIC_LLM_MODEL` | 解题推理模型标识（若该模型原生支持视觉，VLM 将自动共用同一端点） | 逐题 `stepLogits` / `pitfallLogits` 生成与提示 |
| `EXPO_PUBLIC_VLM_MODEL` | （可选）专职视觉识别模型标识（默认复用上面 `BASE_URL` 与 `KEY`，也可 `EXPO_PUBLIC_VLM_BASE_URL` 单独覆盖） | 拍照/相册公式识题 |
| `EXPO_PUBLIC_EDGE_LLM_URL` | （可选）本地或局域网模型覆盖地址（如 `http://auto:11434` 自动跟随 Metro IP） | 隐私离线本地 Ollama 推理 |
| `EXPO_PUBLIC_AGENT_BACKEND_URL` | （可选）个性化学习智能体 FastAPI 后端地址 | 跨端遥测归档同步与长期学情教练报告 |

> 说明：Expo 会将 `EXPO_PUBLIC_*` 打入客户端 JS 包中，公开分发的正式版应留空密钥并通过后端代理鉴权（详见 [FULLSTACK_DEV_REQUIREMENTS.md](docs/FULLSTACK_DEV_REQUIREMENTS.md) 第 `BR-03` 节）。

## 导出和查看 JSON（v2.0-annotated）

点「导出」可将当前会话保存至应用文档目录，也可「另存到手机文件夹」。导出除 `strokes` 与 `stallEvents` 外，还包含伴随微标注 `annotations`、降采样三维状态及图谱进度轨迹 `stateTrace`（新增 `anchorGap` 与 `stepProgress`）、过程性压缩遥测 `learningTelemetry`（`agent-telemetry-v1`）与本地数字孪生 `learnerTwin`（`twin-v1`），标记为 `meta.version = "2.0-annotated"`，可用于后端分析与 `ml/` 模型微调验证。

```json
{
  "meta": { "exportedAt": "2026-01-01T00:00:04.000Z", "strokeCount": 2, "pointCount": 3, "version": "2.0-annotated" },
  "strokes": [
    {
      "id": 1,
      "points": [
        { "x": 44, "y": 96, "t": 1767225600000 },
        { "x": 58, "y": 96, "t": 1767225600017 }
      ]
    },
    { "id": 2, "tool": "eraser", "points": [{ "x": 58, "y": 96, "t": 1767225600200 }] }
  ],
  "stallEvents": []
}
```

## 端侧神经架构、最新实测与智能体进展

### 1. 已落地的全面升级架构

本分支已完成从“启发规则 + 6 通道小模型”向**图谱闭环神经状态机 + 个性化学习智能体**的升级（Expo Go 保持零新增原生包，纯 TypeScript `Float32Array` 内核开箱即跑）：

- **流 1 · 10 通道图关系感知识别器 (`EdgeMathGraphNet`)**：
  - **Quote ① 空间拓扑重排** (`canonicalizeLineStrokesForNeural` / Python `canonicalize_strokes_xy`)：按分式线上下簇与 `\int`/`\sum` 界限簇进行二维规范排序，解除回头补写造成的纯时间乱序；
  - **Quote ③ 10 通道编码 + 7 类笔画边关系头**：在 `[dx, dy, cosθ, sinθ, pen, y_rel]` 基础上新增 4 维笔画归一化包络 `[st_cx_rel, st_cy_rel, st_w_norm, st_h_norm]`（491.0K 参数），配合双线性 `StrokeGraphEdgeHead`（17.1K 参数）预测 `NONE / SAME_SYMBOL / RIGHT / ABOVE / BELOW / SUP / SUB` 7 类边关系辅助多笔符号聚簇；
  - **Quote ② 题谱先验浅融合 + TAMER 语法束搜索**：根据当前题干及解题图谱 `expectedTokens` 施加对数概率先验偏置，并在 Beam Search 中加入括号/上下标未闭合惩罚及收尾修复（`repairBalancedTokens`）；
  - **Quote ④ 可选 ONNX Coverage 解码加速轨**：训练并导出基于卷积覆盖记忆的 `CoverageMathSeq2SeqDecoder`（`0.65M` 参数）与 [`OnnxEdgeSyntaxAdapter`](src/engine/onnxSyntaxAdapter.ts)，在 Dev Client / Release APK 接入 `onnxruntime-react-native` 即可开启，Expo Go 自动无缝降级回纯 TS INT8 核。
- **图谱动态 Logits 验证闭环（图二）**：不再依赖固定 4 条易错 `if/else` 规则；由商业大模型随题目动态生成 `steps[].stepLogits` 与 `commonPitfalls[].pitfallLogits`（或由 [`graphLogits.ts`](src/engine/graphLogits.ts) 规整/兜底编译），与 `P_ctc(v)` 边缘后验及 7 类几何角色激活执行连续软概率内积，适用于任意微积分、线代与概率统计新题。
- **14 维因果状态机闭环（图四）**：`CognitiveStateNet` 动力学流由 10 维扩展至 **14 维**（新增 `anchorGap`、`stepStagnation`、`stepAdvancePulse`、`stepProgress`），状态机引入跨步心算保护盾 `stepJumpShield`、滞后阻力叠加 `stagnationStress` 与停滞接管 `graphStepStagnationStall`，结合 7 维上下文 LinUCB 实现 6 风格 `L0-L3` 路由。
- **个性化学习智能体端侧闭环 (`src/agent/`)**：在切题与导出时自动调用 `finalizeCurrentSession()`，将过程性压缩指标（`agent-telemetry-v1`，不含任何原始坐标）喂给 `ProcessKnowledgeTracer` 执行连续软观测 **Process-BKT**，维护跨题习惯数字孪生（`inkpulse-learner-twin.json`）并支持「独立后端 → 端侧直连商用 LLM → 本地离线处方」三级教练报告生成。

### 2. 目前实测基准与工程状态

| 模块 | 当前状态 | 实测关键指标与说明（RTX 5060 CUDA + Pure-TS 验证） |
| :--- | --- | :--- |
| **10 通道 `EdgeMathGraphNet` + `StrokeGraphEdgeHead`** | **已完成训练与双端量化** | 基于 Google MathWriting 2024（`32,000` 训练 / `1,600` 验证 / `4,000` 留出测试 + `2,000` 笔画边图样本）训练 20 Epoch：`real_val_CER = 0.2571`、`Edge_Val_Acc = 98.61%`；4K 留出真实测试集 `Greedy_CER = 0.2741`、`TAMER Beam4 CER = 0.2771`。TS INT8 权重包 [edgeNeuralWeights.ts](src/engine/edgeNeuralWeights.ts) 体积 **`832.4 KB`**；PyTorch ↔ Pure-TS 10 通道语义最大差 `0.1141`、EdgeHead 概率差 `0.0071`（`cls_match=3/3`），单次 TS 编码平均约 `26.8 ms` |
| **可选 `CoverageMathSeq2SeqDecoder`** | **PyTorch/ONNX 已导出校验** | 参数 `0.65M`（6 Epoch `ce_loss = 0.5189 → 0.1498`），导出 `ml/runs/onnx/edge_math_coverage_decoder.onnx` 体积 `2534.7 KB`（ORT 最大差 `4.77e-06`）；[OnnxEdgeSyntaxAdapter](src/engine/onnxSyntaxAdapter.ts) 已落地并兼容 Expo Go 回退 |
| **14 维图谱融合 `CognitiveStateNet`** | **已贯通闭环；待真人数据微调** | PyTorch ↔ ORT 误差 `1.43e-06`，PyTorch ↔ Pure-TS 6 分类预测一致 `6/6`（`prob_diff = 0.0000`、`mra_diff = 0.0012`），单窗耗时 **`1.4~1.5 ms`**。注：目前 Stage 3 使用合成标注会话训练，后续需采集真实人类 `2.0-annotated` 数据替换并跑 LOUO-CV |
| **图谱动态 Logits 与 `src/agent/`** | **端侧核已贯通，UI/后端见 PRD** | 任意自定义新题 `pitfallLogits`、`graphStepStagnationStall`、Process-BKT 与 `offline_heuristic` 教练建议均通过 Node 端到端断言，`npm run typecheck` 零报错 |

### 3. 下一步重点开发需求

完整的前后端任务拆解、API 契约与 PostgreSQL DDL 请见 [前后端完整开发需求文档](docs/FULLSTACK_DEV_REQUIREMENTS.md) 与 [智能体架构说明书](docs/LEARNING_AGENT_SPEC.md)：

1. **后端服务开发与部署**：落地 FastAPI + PostgreSQL + Redis 服务，提供商用大模型鉴权代理与 `solution_graph_cache` 云端共享图谱缓存（大幅削减同题 API 开销）、幂等遥测入库、跨周教练记忆检索与匿名标注样本收集。
2. **移动端智能体 UI 与交互完善**：新增「学情孪生与诊断报告」页面组件（展示 8 大考点掌握度、书写习惯仪表盘与 3 级降级 AI 教练建议），并在实时诊断区展示步骤推进度与锚点缺口。
3. **真实人类采样数据微调 Stage 3**：收集真实学生多速度、多设备 `2.0-annotated` 会话，通过 `ml/train/evaluate_real_louo.py` 验证并替代合成会话微调 14 维 `CognitiveStateNet`。
4. **Dev Client / APK ONNX 原生开启**：在独立打包配置中按需启用 `onnxruntime-react-native` 与模型文件下载校验以释放 Coverage 解码精度。

## 如何打包成 APK

项目配置的安卓包名为 `app.inkpulse.notebook`，应用版本为 `1.2.0`，安卓版本号为 `3`。`eas.json` 中的 `development` 配置用于包含设备端原生模块的开发构建，`preview` 配置指定内部安装与 APK 构建。

1. 在已安装依赖的仓库根目录安装 EAS CLI，使用自己的 Expo 账号登录并确认当前账号。

   ```sh
   npm install --global eas-cli
   eas login
   eas whoami
   ```

2. 首次构建执行 `eas init`，按提示创建或关联自己的 Expo 项目。CLI 会把该项目的标识写入 `app.json`。公开演示包可直接使用离线提示与离线例题图谱；仅限受控演示时，可在 Expo 项目的 **Environment variables → preview** 中设置前述环境变量（EAS 构建服务器不会上传本机 `.env.local`）。
3. 从仓库根目录提交安卓构建，首次按 EAS 提示创建或选择 Android 签名密钥。之后发布同包名的更新请沿用同一签名。

   ```sh
   eas build --platform android --profile preview
   ```

4. 打开终端给出的构建详情链接，等待状态变为 **Finished**，点下载获取 **`.apk` 安装文件**。也可以运行 `eas build:list --platform android` 找回记录。[Expo 首次构建指南](https://docs.expo.dev/build/setup/) · [Expo 安卓 APK 指南](https://docs.expo.dev/build-reference/apk/)

## 验收用例

| 验证内容 | 操作与预期结果 |
| --- | --- |
| 题干 VLM 识题与重识 | 配置商用 VLM 变量，点击「题干」→「拍照/相册识题」并裁剪。识别结果填入预览；若因网络失败可点「用原裁剪图重识」而无需重新拍图；全程不混入无用文本报错 |
| 动态图谱预解与提示 | 粘贴任意大学高数/线代/概率题目点击完成，确认后台预解成功（状态变为「图谱已缓存」）；触发卡壳后 `L0-L3` 提示遵循步骤方向，均不直接泄漏 `sealedFinalAnswer` |
| 离线图谱与多轨降级 | 清空 API Key 或断网重载后选择 6 道内置样例题或自由做题，确认 10 通道公式编码、图谱 Logits 判别与离线风格提示均可正常工作 |
| 书写、橡皮与撤销/重做 | 写两笔后用橡皮擦除一部分，点「撤销」恢复本次擦除；再次撤销最后一笔消失；点「重做」恢复；撤销后重新落笔重做栈清空；清空画布同步重置追踪状态 |
| 心算保护与停滞判定 | 写完一步核心等式后进行短暂心算停顿，确认不会立即误报；连续在同一步长期无进展停笔或触发高概率易错歧路时，状态机正常触发引导 |
| 诊断、孪生更新与导出 | 查看「采样诊断」滑块生效与「历史」记录；点击「导出」检查生成的 `2.0-annotated` JSON 中同时包含 `annotations`、`stateTrace`、`learningTelemetry` 与 `learnerTwin` |
| 屏幕适配 | 在手机竖屏、平板横竖屏打开题干键盘、识题结果、诊断、提示、历史、导出，确认布局可滚动、按钮可触及且旋转后可继续书写 |

## 真机性能记录

早期采集版本在 OPD2506 平板（Android 16、Expo Go 58.0.5）上对每个完整一秒窗口逐一统计。120 Hz 自动连续划线 12.790 秒得到 115–122 点/秒；60 Hz 同动作 12.785 秒得到 59–62 点/秒；默认模式真人书写的两条长笔最低分别为 57、53 点/秒。这些数字描述触点送达速率；屏幕刷新率、实际渲染帧率和进程内存需要分别测量。

该平板报告的处理器型号为 MT8799，物理屏幕为 2120 × 3000，Android 内核可见总内存约 7.40 GiB（`MemTotal` 7,762,248 kB）。

| 版本 / 运行方式 | 设备 / Android | 处理器 | 物理内存 | 屏幕分辨率 / 刷新率 | 实际渲染帧率 | 书写后内存 PSS |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

## 代码与开源组件

- `src/capture/` 使用开源 Gesture Handler、Reanimated 与 Worklets 在 UI 线程接收触点，以单调时钟映射 Unix 毫秒；Skia 绘图随帧刷新，采样与显示分离。
- `src/stalls/` 提供基础几何/停顿/速度特征滑动窗口与事件分组。
- `src/engine/` 端侧多源认知与图谱推理引擎：包含 10 通道重采样与 7 类边关系推理（`neuralRuntime.ts`）、空间重排与连续图谱 Logits 评测（`inkSyntaxEncoder.ts` & `graphLogits.ts`）、14 维因果认知状态机与个人运动指纹（`cognitiveState.ts`）、7 维 LinUCB 个性化路由（`personalizedPolicy.ts`）、商用 VLM/LLM 统一接入与动态 Logits 图谱缓存（`problemIngestor.ts` & `solutionCache.ts`）、可选 ONNX 原生适配轨（`onnxSyntaxAdapter.ts`）、端侧提示合成（`edgeHintModel.ts`）与伴随微标注采集（`annotationTracker.ts`）。
- `src/agent/` 个性化学习智能体端侧模块：过程遥测有界压缩聚合（`sessionAggregator.ts`）、Process-BKT 知识追踪与习惯数字孪生（`processKnowledgeTracer.ts`）以及三级降级教练客户端（`coachClient.ts`）。
- `src/hints/` 管理防剧透校验 `filterSafeHintText`、提示分层服务与多风格文案路由。
- `src/ocr/` 集成 `expo-image-picker`、`expo-image-manipulator` 进行裁剪与压缩，优先调用统一商用 VLM API，仅在特定条件下按需回退本机 `rn-mlkit-ocr`。
- `src/export/` 基于 `expo-file-system` 原子写入 `2.0-annotated` 会话快照与本地持久化存储。
- `ml/` 提供 PyTorch CUDA 10 通道 `EdgeMathGraphNet` + `StrokeGraphEdgeHead` + `CoverageMathSeq2SeqDecoder` + 14 维 `CognitiveStateNet` 四阶段训练、ONNX 导出（ORT 对齐）与 INT8 TypeScript 量化打包管线。
- `docs/` 收录 [FULLSTACK_DEV_REQUIREMENTS.md](docs/FULLSTACK_DEV_REQUIREMENTS.md) 与 [LEARNING_AGENT_SPEC.md](docs/LEARNING_AGENT_SPEC.md)。

从 `main` 创建分支，并用中文写明设备、输入动作及验证结果。[贡献指南](CONTRIBUTING.md) · [MIT 许可证](LICENSE)
