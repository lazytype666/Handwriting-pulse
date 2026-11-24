# 《笔迹脉搏》前后端完整开发需求文档（Dev PRD）

> 本文档与系统技术架构说明书 [LEARNING_AGENT_SPEC.md](LEARNING_AGENT_SPEC.md) 配套使用：后者描述端侧智能体算法与数据协议，本文档面向**前端（React Native / Expo）、后端（FastAPI / PostgreSQL）、算法工程**团队明确下一阶段的开发需求边界、接口契约、交互设计及上线验收标准。

---

## 一、总体架构与当前工程基线

### 1.1 系统角色分工

| 分层 | 技术栈 | 当前状态 | 核心定位 |
| :--- | :--- | :--- | :--- |
| **前端移动端** | React Native + Expo + TypeScript (`src/`, `App.tsx`) | **端侧引擎与智能体 SDK 已就绪；智能体展示 UI 待开发** | 笔迹采集、零原生依赖纯 TS INT8 推理 / Dev Client ONNX 加速、14 维状态机实时触发、过程遥测与 Process-BKT 孪生本地持久化 |
| **商业大模型云脑** | OpenAI 兼容多模态/推理 API（[.env.example](../.env.example)） | **端侧直连调用已就绪；生产期建议经后端代理鉴权** | 拍题/选图 VLM 识题、逐题生成动态 `stepLogits` / `pitfallLogits` 解题图谱、生成个性化思维与书写习惯教练报告 |
| **独立智能体后端** | FastAPI + PostgreSQL 16 + Redis（契约见 [LEARNING_AGENT_SPEC.md](LEARNING_AGENT_SPEC.md)） | **前端异步客户端已对接；后端服务待开发部署** | 多端数字孪生同步、跨周学情记忆聚合、题目图谱共享缓存降本、API Key 安全托管、脱敏标注样本回流训练池 |
| **算法训练与导出管线** | PyTorch CUDA + ONNX + INT8 (`ml/`) | **10 通道图网络 + Coverage 解码器 + 14 维状态机已落地** | `EdgeMathGraphNet` + `StrokeGraphEdgeHead` + `CoverageMathSeq2SeqDecoder` + `CognitiveStateNet` 训练、ORT 校验与端侧权重打包 |

### 1.2 已落地的核心端侧能力清单（后端与 UI 直接复用）
1. **统一 VLM + LLM 识题与图谱预解**：[`ProblemIngestorAdapter`](../src/engine/problemIngestor.ts)、[`captureProblemForIngest`](../src/ocr/index.ts)、[`SolutionCacheManager`](../src/engine/solutionCache.ts) 与 [`graphLogits.ts`](../src/engine/graphLogits.ts)。
2. **端侧公式识别 + 图关系推理 + 14 维状态机**：[`EdgeNeuralModelRunner`](../src/engine/neuralRuntime.ts)、[`EdgeInkSyntaxEncoder`](../src/engine/inkSyntaxEncoder.ts)、[`CognitiveStateEngine`](../src/engine/cognitiveState.ts)、[`OnnxEdgeSyntaxAdapter`](../src/engine/onnxSyntaxAdapter.ts)。
3. **个性化学习智能体端侧闭环**：[`SessionTelemetryAggregator`](../src/agent/sessionAggregator.ts)、[`ProcessKnowledgeTracer`](../src/agent/processKnowledgeTracer.ts)、[`LearningCoachAgentClient`](../src/agent/coachClient.ts) 及 [`LocalStorageAdapter`](../src/engine/storageAdapter.ts)（已在 [`App.tsx`](../App.tsx) 的切题 `onSave` 与导出 `exportInk` 时自动归档更新 `inkpulse-learner-twin.json`）。

---

## 二、前端（移动端 App）开发需求

### FR-01：个性化学习智能体与「学情孪生」中心页面（P0 · 核心新增）
- **交互入口**：在 [`App.tsx`](../App.tsx) 顶部导航栏（[`HistoryView.tsx`](../src/ui/HistoryView.tsx) 按钮旁）新增「学情报告」入口，弹窗或抽屉展示 `CoachAgentView` 页面。
- **数据源对接**：打开页面时直接读取 `engineRef.current.getLearnerTwin()` 与 `engineRef.current.buildSessionTelemetrySnapshot()`。
- **界面模块拆解**：
  1. **概览卡**：显示累计做题数 `totalSessions`、完成题数 `totalProblemsSolved`、当前主导短板中文标签 `dominantWeakness`、系统偏好提示风格 `preferredHintStyle`。
  2. **知识点掌握度分布（Process-BKT）**：遍历 `twin.conceptMastery`，按 `masteryProb` 从低到高渲染 8 大大学数学考点掌握进度条、练习次数 `attempts` 与歧路命中数 `pitfallHits`；优先高亮最薄弱 Top 3 考点。
  3. **书写与思辨习惯仪表盘**：可视化 `habitProfile` 核心指标——平均书写动量 `avgMomentum`、单笔涂改/撤销率 `avgEraseRatio`、结构边界停顿占比 `structuralPauseRatio`、深度心算静音占比 `deepThinkingRatio`、歧路触发频率 `pitfallFrequency`。
  4. **AI 学习教练诊断卡片**：点击「生成本次思维与习惯诊断」触发 `engineRef.current.requestCoachAdvice()`（加载态 + 超时提示），展示 `summary`、`thinkingDiagnosis`、`habitDiagnosis`、逐条可勾选 `nextActionPlan`，并显示当前诊断来源徽标（`云端智能体后端` / `端侧直连大模型` / `离线本地引擎`）。
- **验收标准**：断网情况下页面 100% 正常打开且「生成诊断」可返回 `offline_heuristic` 完整建议；有网且配置商用 API 或后端时返回个性化诊断。

### FR-02：做题过程中的图谱对齐微状态反馈（P1）
- **复用组件**：[`DiagnosticsPanel.tsx`](../src/ui/DiagnosticsPanel.tsx) 与 [`HintCard.tsx`](../src/ui/HintCard.tsx)。
- **需求描述**：
  1. 在诊断面板「实时特征」展开区补充展示图谱对齐指标：`matchedStepIndex + 1`（当前匹配步骤）、`stepProgress`（百分比推进度）、`anchorGap`（当前步缺失关键符号缺口）与 `pitfallProb`（实时歧路概率），方便教研调试与高阶自查。
  2. 严禁在前端任何组件或调试浮层渲染 `SolutionCacheEntry.sealedFinalAnswer`。

### FR-03：识题失败重试与模型配置引导（P1）
- **复用组件**：[`QuestionEditor.tsx`](../src/ui/QuestionEditor.tsx)。
- **需求描述**：
  1. 完善「用原裁剪图重识」触发时的加载禁用状态与耗时提示，若连续两次失败引导用户直接手写修正或粘贴纯文本题干。
  2. 在生产版本中去除前端直连 `EXPO_PUBLIC_LLM_API_KEY` 明文打包风险，增加「已切换至托管网关」连接状态指示（见 `BR-03` 后端代理需求）。

### FR-04：Dev Client / Release APK ONNX 原生加速与模型热下发（P2）
- **对接模块**：[`OnnxEdgeSyntaxAdapter`](../src/engine/onnxSyntaxAdapter.ts)、[`HandwritingPulseEngine.registerSyntaxEncoderAdapter`](../src/engine/index.ts)。
- **需求描述**：
  1. 仅在自定义 Dev Client / Release APK 构建脚本中添加 `onnxruntime-react-native`（Expo Go 目标保持零原生依赖可运行）。
  2. App 首次启动检测非 Expo Go 环境时，支持从静态 CDN / 后端下载并按 SHA-256 校验 `edge_math_encoder.onnx`（1.97 MB）与 `edge_math_coverage_decoder.onnx`（2.53 MB），通过 [`registerOnnxNativeRuntime`](../src/engine/onnxSyntaxAdapter.ts) 注入后将适配器注册给引擎，解锁长公式自回归 Coverage 解码。

### FR-05：脱敏标注样本授权贡献开关（P2）
- **复用组件**：[`ExportDialog.tsx`](../src/ui/ExportDialog.tsx)。
- **需求描述**：在导出笔迹快照对话框中提供可选勾选框「同步匿名化过程标注至研究服务器（不含个人身份，用于改进停顿与公式识别模型）」，确认后上传至 `BR-05`。

---

## 三、后端开发需求（FastAPI + PostgreSQL + Redis）

> 后端接口路径与基础表结构需严格兼容 [`LearningCoachAgentClient`](../src/agent/coachClient.ts) 已对接的契约与 [LEARNING_AGENT_SPEC.md](LEARNING_AGENT_SPEC.md) 第 4 节。

### BR-01：用户/设备身份与鉴权体系（P0）
- **接口**：
  - `POST /v1/auth/device-session`：接收端侧匿名 `device_uuid`，签发短期 JWT + 刷新令牌，初始化或返回对应 `user_id`。
  - `POST /v1/auth/bind`：支持后续手机号/学号登录将匿名设备历史 `learner_twins` 与 `learning_sessions` 合并至正式账号。
- **安全要求**：生产环境所有 `/v1/telemetry/*`、`/v1/agent/*`、`/v1/graph/*` 均需校验 `Authorization: Bearer <token>`，并针对单设备配置滑动窗口限流（防止 LLM 网关刷量）。

### BR-02：过程性遥测入库与幂等数字孪生更新（P0）
- **接口**：`POST /v1/telemetry/session`
- **输入**：`{ userId?, telemetry: SessionLearningTelemetry, twin: LearnerCognitiveTwin }`
- **业务逻辑**：
  1. Pydantic 强校验 `telemetry.schemaVersion == "agent-telemetry-v1"`，显式丢弃任何未声明字段（确保绝不意外留存原始坐标 `(x,y,t)`）。
  2. 以 `session_id` 主键幂等插入 PostgreSQL `learning_sessions`（`ON CONFLICT (session_id) DO NOTHING`）。
  3. 仅当传入为新会话时，在事务内更新 `learner_twins` 表的 `total_sessions`、`total_solved`、`dominant_weakness`、`habit_profile` 与 `concept_mastery`。
- **响应**：`200 OK` `{ "ok": true, "storedSessionId": "...", "serverTwinVersion": 12 }`

### BR-03：商业大模型安全代理与云端解题图谱共享缓存（P0 · 降本防泄露）
- **背景**：当前商业 LLM Key 通过 `.env.local` 配置在移动端容易被逆向提取，且相同大学高数/线代典型题会被不同学生重复调用大模型预解图谱。
- **接口**：
  1. `POST /v1/graph/prewarm`：接收规范化题干 `problemText`（与可选图片 Base64 进行 VLM 识题）。
     - 先计算 `problem_hash = sha256(normalizeProblemKey(problemText))` 查询 Redis + PostgreSQL `solution_graph_cache`；
     - 命中则 20ms 内直接返回带动态 `stepLogits`、`pitfallLogits`、`hintsByStyle` 与密文/封存 `sealedFinalAnswer` 的标准 `SolutionCacheEntry`（累计节省 70%+ 重复题目商业 API 费用）；
     - 未命中时由后端服务端持有的商业 VLM/LLM API Key 请求生成、经服务端 JSON 校验与 125 词表规整后回写缓存并返回前端。
  2. `POST /v1/vlm/recognize`：代理商业 VLM 图片识题，隐藏前端 Key 并执行图片尺寸与频率限制。

### BR-04：个性化学习智能体长期诊断与跨端恢复（P0）
- **接口**：
  1. `POST /v1/agent/coach-report`：接收端侧上传的当前 `twin` 与 `latestTelemetry`，结合 PostgreSQL 最近 14 天内该用户同考点会话记录与上周习惯均值对比斜率（如 `avgEraseRatio` 周环比变化、`structuralPauseRatio` 变化），拼装紧凑 Prompt 请求后端商业 LLM 生成结构化 `{ summary, thinkingDiagnosis, habitDiagnosis, nextActionPlan }`。
  2. `GET /v1/twin/{user_id}`：多端登录或重装 App 时拉取完整 `LearnerCognitiveTwin` 与最近 20 条 `recentSessions` 摘要。

### BR-05：脱敏会话回流收集与持续训练任务（P1）
- **背景**：当前 14 维 `CognitiveStateNet` 在缺乏真实人手 `2.0-annotated` 标注时使用合成会话验证；后端需打通真实做题标注闭环。
- **接口与离线任务**：
  1. `POST /v1/dataset/annotated-session`：仅在用户明确开启 `FR-05` 授权后接收完整导出 `2.0-annotated` JSON（自动剥离任何题干中可能包含的姓名/班级文本），存储至对象存储 / `dataset_contributions` 表。
  2. 每周定时脚本导出合规标注包至训练机执行 `ml/train/train_pipeline.py --stage cognitive_finetune --data-dir ...`，产出新版 `CognitiveStateNet` 后通过 `ml/export/export_onnx.py` 与 `export_ts_weights.py` 生成权重更新包，同时用历史 `learning_sessions` 对 8 个考点的 Process-BKT 参数（$p(\text{init}), p(\text{slip}), p(\text{guess})$）做 EM 校准。

---

## 四、补充数据库表设计（PostgreSQL）

在 [LEARNING_AGENT_SPEC.md](LEARNING_AGENT_SPEC.md) 已定义的 `learner_twins`、`learning_sessions` 两张核心表基础上，补充后端图谱共享缓存与标注回流表：

```sql
-- 1. 云端解题图谱与动态 Logits 共享缓存表（大幅降低商业 LLM 重复调用成本）
CREATE TABLE IF NOT EXISTS solution_graph_cache (
  problem_hash        CHAR(64) PRIMARY KEY,
  normalized_text     TEXT NOT NULL,
  steps_json          JSONB NOT NULL,      -- 含 stepLogits
  pitfalls_json       JSONB NOT NULL,      -- 含 pitfallLogits
  hints_by_style      JSONB NOT NULL,
  sealed_final_answer TEXT,
  model_tag           TEXT NOT NULL,
  hit_count           INTEGER NOT NULL DEFAULT 1,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_solution_cache_hits
  ON solution_graph_cache (hit_count DESC, updated_at DESC);

-- 2. 用户授权贡献的 2.0-annotated 微标注训练样本索引表
CREATE TABLE IF NOT EXISTS dataset_contributions (
  contribution_id     TEXT PRIMARY KEY,
  user_id             TEXT REFERENCES learner_twins(user_id),
  stroke_count        INTEGER NOT NULL,
  annotation_counts   JSONB NOT NULL,
  payload_object_key  TEXT NOT NULL,       -- S3 / MinIO / OSS 脱敏文件路径
  quality_passed      BOOLEAN NOT NULL DEFAULT FALSE,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 五、非功能性需求、安全合规与交付验收标准

| 维度 | 量化要求 / 红线规则 | 验证手段 |
| :--- | :--- | :--- |
| **隐私安全边界** | 常规智能体业务接口（`/v1/telemetry/session`、`/v1/agent/coach-report`）**严禁接收或持久化原始 `(x, y, t)` 笔迹坐标**；仅在用户显式勾选授权时允许上传去身份 `2.0-annotated` 训练包 | 后端 Pydantic 严格模式 (`extra="forbid"`) + 抓包审计 |
| **防剧透红线** | `steps[].goalDescription`、`pitfalls[].warningHint`、`hintsByStyle`、`coach-report` 及一切 UI 禁止出现 `sealedFinalAnswer` | [`filterSafeHintText`](../src/hints/safety.ts) 端云双重校验 |
| **Expo Go 兼容性** | 主分支 `package.json` 不引入会破坏 Expo Go 沙盒的必选原生崩溃模块；ONNX 加速轨仅作可选注入 | `npm run typecheck` + Expo Go 真机冷启动无红屏 |
| **多轨容灾降级** | 独立后端不可达 → 自动降级前端直连商用 LLM API → 无网/无 Key 自动降级纯端侧 `offline_heuristic` 与静态样例图谱 | 断网与后端 500 故障注入测试 |
| **端侧实时推理性能** | 手机端 JS 侧 `EdgeNeuralModelRunner` 单次 14 维认知状态推断 $\le 10\text{ ms}$，10 通道公式编码 $\le 80\text{ ms}$，不掉帧阻塞画布笔迹采样 | Node/真机性能采样与 60fps 笔迹延迟观测 |
| **数值一致与回归** | 每次更新 `ml/` 模型权重重新导出后，必须通过 PyTorch vs ONNX vs INT8 TS 数值对齐测试（状态概率最大差 $< 0.08$、EdgeHead 概率差 $< 0.08$） | 自动化 parity 检验脚本 + `tsc --noEmit` |
