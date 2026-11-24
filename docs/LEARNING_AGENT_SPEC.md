# 《笔迹脉搏》个性化学习智能体 前后端与端云协同技术说明书

> 版本：`agent-spec-v1.0` | 状态：端侧智能体核心引擎与同步客户端已完整落地 `src/agent/` 与 `src/engine/`；独立云端 FastAPI + PostgreSQL 持久化后端按本文档第 4～5 节契约部署。

---

## 一、设计目标与端云职责边界

针对大学高等数学、线性代数与概率统计做题场景，个性化学习智能体旨在基于**零隐私坐标泄露的过程性手写遥测**，为答题者持续诊断以下两类高价值成长指标：
1. **思维瓶颈（Thinking Diagnosis）**：在何类考点（如 `linalg.matrix_det` 含参行列式临界点、`calculus.limit_taylor` 泰勒展开阶数、`prob.bayes_conditional` 全概率分母权重）常出现**结构边界长停**与**歧路漏讨论**。
2. **书写与运算习惯（Habit Diagnosis）**：包括动量均值 `avgMomentum`、单笔平均涂改/撤销率 `avgEraseRatio`、长停发生在符号运算边界后的占比 `structuralPauseRatio`、深度静默心算占比 `deepThinkingRatio` 与主导短板分类 `dominantWeakness`。

| 层级 | 部署位置 | 已实现 / 规划 | 核心职责 | 模型与数据规模 |
| :--- | :--- | :--- | :--- | :--- |
| **L1 端侧高频感知** | 手机端 App (`src/engine/`) | **已落地** | 10 通道公式识别 + 7 类笔画边图关系 + 每题连续 Logits 步骤/歧路检测 + 14 维因果认知状态机 | `EdgeMathGraphNet` (491.0K) + `CognitiveStateNet` + INT8 TS Bundle (832.4 KB)；可选 `edge_math_coverage_decoder.onnx` (0.65M / 2.53 MB ONNX) |
| **L2 端侧智能体孪生** | 手机端 `src/agent/` + `LocalStorageAdapter` | **已落地** | 过程遥测有界压缩、Process-BKT 连续知识追踪、EMA 书写习惯数字孪生落盘 (`inkpulse-learner-twin.json`) | 零神经网络额外耗时（纯增量统计 + 贝叶斯闭式更新 < 1 ms/题） |
| **L3 商用 VLM/LLM 云脑** | 统一商业 API (`EXPO_PUBLIC_LLM_*`) | **已落地** | 题目图片多模态识别（统一复用 LLM Endpoint）、动态生成 per-problem `stepLogits` / `pitfallLogits`、端云直连教练处方 | 商用多模态/推理模型 API（单题图谱预解 JSON ~ 1–2 KB，教练处方 JSON ~ 0.6 KB） |
| **L4 个性化智能体后端** | 独立服务 (`EXPO_PUBLIC_AGENT_BACKEND_URL`) | **契约已对接，服务端待部署** | 多终端跨周趋势聚合、题目推荐图谱、长期记忆向量检索、BKT 全局参数校准与教师端学情看板 | FastAPI + PostgreSQL 16 (+ pgvector 可选) |

---

## 二、前端（端侧）已落地架构与接口说明

### 2.1 目录与核心模块清单

| 文件路径 | 核心类 / 导出 | 职责说明 |
| :--- | :--- | :--- |
| `src/agent/sessionAggregator.ts` | `SessionTelemetryAggregator`, `extractKnowledgeTags`, `SessionLearningTelemetry`, `WritingHabitMetrics` | 在 60ms～125ms tick 及 `onStrokeComplete`/`onStrokeUndo` 过程中零拷贝统计停顿分布、Z-Score 均值/P90、结构边界长停比、涂改撤销比及最多 24 条关键里程碑 (`representativeEvents`)，不上传原始坐标点序列。 |
| `src/agent/processKnowledgeTracer.ts` | `ProcessKnowledgeTracer`, `createInitialLearnerTwin`, `LearnerCognitiveTwin`, `ConceptMasteryState` | 实现 **Process-Aware BKT**：用 `s_obs = clamp(0.46*progress + 0.30*alignNorm + 0.24*(1-stuckFrac) - pitfallPenalty)` 替代传统非对即错二元观测，同步迭代 8 个高数/线代/概率考点掌握度 `masteryProb` 及 `WritingHabitProfile` EMA 画像（输出 5 类 `dominantWeakness`）。 |
| `src/agent/coachClient.ts` | `LearningCoachAgentClient`, `LearningCoachAdvice` | 三级降级诊断客户端：① 优先调用 `${EXPO_PUBLIC_AGENT_BACKEND_URL}/v1/agent/coach-report`；② 未部署后端时复用 `EXPO_PUBLIC_LLM_BASE_URL`/`KEY`/`MODEL` 生成云端诊断报告；③ 断网时基于本地孪生合成确定性中文处方。 |
| `src/engine/storageAdapter.ts` | `LocalStorageAdapter.loadLearnerTwin` / `saveLearnerTwin` | 通过 `expo-file-system/legacy` 原子写入 `inkpulse-learner-twin.json` 持久化跨题数字孪生。 |
| `src/engine/index.ts` & `App.tsx` | `HandwritingPulseEngine.finalizeCurrentSession` / `requestCoachAdvice` / `getLearnerTwin` | 切换新题（`QuestionEditor` 保存变更时）或点击「导出」(`exportInk`) 自动触发单题遥测归档 → 更新本地 BKT → 落盘 → 异步 POST 至后端。 |

### 2.2 前端过程遥测压缩协议 (`SessionLearningTelemetry`, schema `agent-telemetry-v1`)

每道题完成后生成的遥测载荷仅约 **1.5 KB ~ 3.5 KB** JSON：
- **基础维度**：`sessionId`, `capturedAt`, `durationMs`, `problemKey`, `knowledgeTags`（自动标注如 `['calculus.limit_taylor']`）, `totalStrokes`, `totalStepsExpected`, `maxMatchedStepIndex`, `finalStepProgress`, `finalAnchorGap`, `meanAlignmentA`, `maxPitfallProb`, `hitPitfalls`。
- **书写习惯矢量 (`habits`)**：
  - 节律：`avgMomentumM`, `avgSpeedRatio`, `pauseMedianMs`, `pauseP90Ms`, `pauseZScoreMean`, `pauseZScoreP90`
  - 修正：`totalEraseCount`, `totalUndoCount`, `erasePerStrokeRatio`
  - 六态分布占比：`fluentDerivationRatio`, `deepThinkingHoldRatio`, `stuckCalculationRatio`, `stuckConceptGapRatio`, `pitfallDriftRatio`
  - 关键思辨特征：`structuralBoundaryPauseRatio`（长停顿落在 `= / \frac / \int / \matrix` 等边界锚点后的比率）
- **交互反馈 (`hintFeedbackSummary`)**：`hintsShown`, `acceptedCount`, `dismissedCount`, `topEffectiveStyle`, `rewardSum`
- **浓缩事件时间轴 (`representativeEvents`)**：上限 24 条关键事件（停顿报警、歧路命中、步骤推进、擦除/撤销、提示反馈）。

---

## 三、端侧状态机 × 解题图谱闭环输入规范（图二 / 图四）

1. **图谱逻辑回归头格式**（由商业 LLM 在 `SolutionCacheManager.prewarm` 中随题目动态生成，并经 `src/engine/graphLogits.ts` 校验与规整）：
   - `steps[].stepLogits`: `{ bias, progressCenter, tokenWeights (125-token subset), roleWeights (7 geometric roles) }`
   - `commonPitfalls[].pitfallLogits`: `{ bias, threshold, minLineCount, lineCountWeight, presentWeights, missingWeights, roleWeights }`
2. **连续概率评估 (`EdgeInkSyntaxEncoder.evaluateGraphLogitHeads`)**：
   - 将神经 CTC 边缘后验 $P_{\text{ctc}}(v)$ 与几何角色识别置信度融合，通过 $\sigma(\beta + w_{\text{line}}\cdot L + \sum w^+ P(v) + \sum w^-(1-P(v)) + \sum w_r \cdot \text{role}_r)$ 输出连续 `pitfallProb`、`anchorGap` 与 `stepProgress`，彻底消除旧版固定 4 条易错点 `if/else` 局限。
3. **注入状态机 (`CognitiveStateEngine` & `HandwritingPulseEngine`)**：
   - `CognitiveState` 新增 `anchorGap`, `stepStagnation`, `stepAdvancePulse`, `stepProgress`, `pitfallProb`；
   - `CognitiveStateNet` 动力学输入升维至 **14 维**（前 10 维节律/对齐 + 后 4 维 `[anchorGap, stepStagnation, stepAdvancePulse, stepProgress]`）；
   - 状态机引入 **`graphStepStagnationStall`** 接管与 **`stepJumpShield`** 刚跨步心算保护盾，实现思维停滞与正常心算的精准区分。

---

## 四、独立个性化智能体后端规范（FastAPI + PostgreSQL）

### 4.1 环境变量与客户端开关

在 `.env.local` 中新增可选配置（留空则自动切换至轨 B 商业大模型直连或轨 C 本地离线处方）：
```ini
EXPO_PUBLIC_AGENT_BACKEND_URL=https://api.your-domain.com/inkpulse
```

### 4.2 REST API 接口契约

1. **`POST /v1/telemetry/session`** — 接收单题完成遥测与前端 BKT 孪生快照
   - Request JSON: `{ "userId": "optional_uid", "telemetry": SessionLearningTelemetry, "twin": LearnerCognitiveTwin }`
   - Response JSON (`200 OK`): `{ "ok": true, "storedSessionId": "sess_...", "serverTwinVersion": 42 }`
   - 幂等性约束：`session_id` 主键去重（`ON CONFLICT (session_id) DO NOTHING`）。

2. **`POST /v1/agent/coach-report`** — 结合多周跨题序列与历史错题倾向生成个性化教练报告
   - Request JSON: `{ "twin": LearnerCognitiveTwin, "latestTelemetry": SessionLearningTelemetry | null }`
   - Server 侧流程：
     1) 查询 PostgreSQL 最近 14 天同知识点 (`knowledge_tags`) 会话聚合统计；
     2) 比较用户当前 `habitProfile` 与历史周均值变化斜率（如涂改率环比下降/升高）；
     3) 调用后端托管的商用大模型（如 DeepSeek-V3 / Qwen-Max / GPT-4o）输出结构化诊断 JSON；
   - Response JSON (`200 OK`):
     ```json
     {
       "summary": "近 5 题微积分均值计算稳健，但含参行列式临界零点仍易漏讨论。",
       "thinkingDiagnosis": "在 linalg.matrix_det 题目中结构边界长停占 68%...",
       "habitDiagnosis": "平均动量 0.64 较上周提升，但单笔涂改率 0.34 集中在移项环节...",
       "nextActionPlan": [
         "遇到含参矩阵题先于右上角写 |A|=0 求临界点再做行变换",
         "草稿区独立完成二次展开后再抄回主区以降低涂改中断",
         "下两题建议先开启「苏格拉底」模式自查每步前提"
       ]
     }
     ```

3. **`GET /v1/twin/{user_id}`** — 多端恢复数字孪生画像 (`LearnerCognitiveTwin`)。

### 4.3 PostgreSQL 数据库 DDL 参考

```sql
CREATE TABLE IF NOT EXISTS learner_twins (
  user_id             TEXT PRIMARY KEY,
  schema_version      TEXT NOT NULL DEFAULT 'twin-v1',
  total_sessions      INTEGER NOT NULL DEFAULT 0,
  total_solved        INTEGER NOT NULL DEFAULT 0,
  preferred_style     TEXT,
  dominant_weakness   TEXT NOT NULL DEFAULT 'balanced',
  habit_profile       JSONB NOT NULL,
  concept_mastery     JSONB NOT NULL,
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS learning_sessions (
  session_id          TEXT PRIMARY KEY,
  user_id             TEXT NOT NULL REFERENCES learner_twins(user_id),
  problem_key         TEXT NOT NULL,
  knowledge_tags      TEXT[] NOT NULL,
  duration_ms         INTEGER NOT NULL,
  total_strokes       INTEGER NOT NULL,
  final_step_progress REAL NOT NULL,
  mean_alignment_a    REAL NOT NULL,
  max_pitfall_prob    REAL NOT NULL,
  hit_pitfalls        JSONB NOT NULL,
  habits              JSONB NOT NULL,
  hint_summary        JSONB NOT NULL,
  representative_evts JSONB NOT NULL,
  captured_at         TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_time
  ON learning_sessions (user_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_tags_gin
  ON learning_sessions USING GIN (knowledge_tags);
```

---

## 五、模型形态与合理规模建议（演进路线图）

| 阶段 | 用户规模 | 模型组合与参数规模 | 单题成本与端云延迟预算 |
| :--- | :--- | :--- | :--- |
| **Stage 1（当前架构·单机/小班测试）** | 1～500 人 | **端侧** `EdgeMathGraphNet` (491K, INT8 832KB) + 14 维 `CognitiveStateNet` + 纯 TS Process-BKT；**云端** 统一商业 LLM API (`EXPO_PUBLIC_LLM_*`) 完成识题/图谱 Logits/直连教练 | 端侧推理 < 15ms；单题云端（1次 VLM 识题 + 1次图谱预解 + 按需1次教练简报）总 Token 约 2.5K～4K |
| **Stage 2（校园/学科推广·端云双轨）** | 500～20,000 人 | **端侧 Dev Client / APK** 启用 `OnnxEdgeSyntaxAdapter` + `CoverageMathSeq2SeqDecoder` ONNX 加速轨（2.53 MB）；**后端** FastAPI 汇总 `learning_sessions` 与全局 BKT 题目难度参数 $p(\text{slip}), p(\text{guess})$ 周期性重估计；高频常见题干做 **Redis + PostgreSQL 图谱缓存复用** | 命中图谱缓存时云端增量开销接近 0；单卡 GPU / FastAPI 即可支撑 200+ QPS 遥测写入与统计查询 |
| **Stage 3（大规模个性化自适应）** | 20,000+ 人 | 后端定期利用脱敏 `learning_sessions` + `stateTrace` 微调轻量 **DKT / Graph-KT (2M～10M 参数)** 用于长期跨知识点迁移推荐，并持续蒸馏端侧 `CognitiveStateNet`（仅下发权重更新包，保持 **零原始笔迹坐标上云** 的隐私架构） | 端侧始终零隐私上传、毫秒级响应；云端专注宏观学情预测与分层次题库调度 |
