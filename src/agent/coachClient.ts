import { resolveChatEndpoint, type CompletionResponse } from '../hints/safety';
import {
  type ConceptMasteryState,
  type LearnerCognitiveTwin,
  type WritingDominantWeakness,
} from './processKnowledgeTracer';
import {
  KNOWLEDGE_TAG_LABELS,
  type KnowledgeConceptTag,
  type SessionLearningTelemetry,
} from './sessionAggregator';

export interface LearningCoachAdvice {
  summary: string;
  thinkingDiagnosis: string;
  habitDiagnosis: string;
  nextActionPlan: string[];
  weakConcepts: Array<{
    tag: KnowledgeConceptTag;
    label: string;
    masteryProb: number;
    pitfallHits: number;
  }>;
  dominantWeakness: WritingDominantWeakness;
  sourceMode: 'agent_backend' | 'commercial_llm_edge' | 'offline_heuristic';
  generatedAt: number;
}

const WEAKNESS_LABELS: Record<WritingDominantWeakness, string> = {
  concept_gap: '定理切换与起手断链倾向',
  calculation_carelessness: '符号化简与高涂改复算倾向',
  boundary_oversight: '临界条件与分式/零点边界漏讨论',
  pace_instability: '作答节奏波动与长停顿焦虑',
  balanced: '思维与落笔节奏总体均衡',
};

function parseCoachJson(raw: string): Partial<{
  summary: string;
  thinkingDiagnosis: string;
  habitDiagnosis: string;
  nextActionPlan: string[];
}> | null {
  const cleaned = raw
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/```\s*$/i, '')
    .trim();
  const s = cleaned.indexOf('{');
  const e = cleaned.lastIndexOf('}');
  if (s < 0 || e <= s) return null;
  try {
    const parsed = JSON.parse(cleaned.slice(s, e + 1));
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * 个性化学习智能体双轨客户端：
 * 1) 若配置了 `EXPO_PUBLIC_AGENT_BACKEND_URL`，优先异步上报过程遥测至智能体后端并请求长期趋势诊断；
 * 2) 未部署独立后端时，无缝复用商用大模型 API（`EXPO_PUBLIC_LLM_*`）生成端边直连诊断报告；
 * 3) 完全离线环境下仍可由数字孪生 `LearnerCognitiveTwin` 本地合成思维+书写习惯改进处方。
 */
export class LearningCoachAgentClient {
  private getBackendUrl(): string | null {
    const url = process.env.EXPO_PUBLIC_AGENT_BACKEND_URL?.trim().replace(/\/+$/, '');
    return url && url.length > 0 ? url : null;
  }

  private getTopWeakConcepts(twin: LearnerCognitiveTwin, topK = 3): ConceptMasteryState[] {
    return Object.values(twin.conceptMastery)
      .sort((a, b) => a.masteryProb - b.masteryProb || b.pitfallHits - a.pitfallHits)
      .slice(0, topK);
  }

  public async syncSessionTelemetry(
    telemetry: SessionLearningTelemetry,
    twin: LearnerCognitiveTwin,
    timeoutMs = 8000,
  ): Promise<boolean> {
    const base = this.getBackendUrl();
    if (!base) return false;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const resp = await fetch(`${base}/v1/telemetry/session`, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ telemetry, twin }),
      });
      return resp.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(timer);
    }
  }

  public async requestCoachReport(
    twin: LearnerCognitiveTwin,
    latestTelemetry?: SessionLearningTelemetry | null,
    timeoutMs = 15000,
  ): Promise<LearningCoachAdvice> {
    const weakList = this.getTopWeakConcepts(twin, 3).map(c => ({
      tag: c.tag,
      label: c.label || KNOWLEDGE_TAG_LABELS[c.tag],
      masteryProb: c.masteryProb,
      pitfallHits: c.pitfallHits,
    }));
    const backendUrl = this.getBackendUrl();

    if (backendUrl) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const resp = await fetch(`${backendUrl}/v1/agent/coach-report`, {
          method: 'POST',
          signal: controller.signal,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ twin, latestTelemetry }),
        });
        if (resp.ok) {
          const data = (await resp.json()) as Partial<LearningCoachAdvice>;
          if (data.summary && data.thinkingDiagnosis && data.habitDiagnosis) {
            return {
              summary: data.summary,
              thinkingDiagnosis: data.thinkingDiagnosis,
              habitDiagnosis: data.habitDiagnosis,
              nextActionPlan: Array.isArray(data.nextActionPlan)
                ? data.nextActionPlan.map(String).slice(0, 4)
                : [],
              weakConcepts: weakList,
              dominantWeakness: twin.habitProfile.dominantWeakness,
              sourceMode: 'agent_backend',
              generatedAt: Date.now(),
            };
          }
        }
      } catch {
        // 回退到端侧直连商用大模型或本地启发处方
      } finally {
        clearTimeout(timer);
      }
    }

    const llmBase = process.env.EXPO_PUBLIC_LLM_BASE_URL?.trim().replace(/\/+$/, '');
    const llmKey = process.env.EXPO_PUBLIC_LLM_API_KEY?.trim();
    const llmModel = process.env.EXPO_PUBLIC_LLM_MODEL?.trim();
    if (llmBase && llmKey && llmModel) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const endpoint = resolveChatEndpoint(llmBase, false);
        const prompt = [
          '你是大学理工数学「思维与书写习惯」私人诊断教练。请依据以下无隐私风险的过程性统计输出严格 JSON：',
          `{ "summary": "50字内总体态势", "thinkingDiagnosis": "结合薄弱考点与结构边界停顿的思维诊断", "habitDiagnosis": "结合涂改率、动量与心算比例的书写习惯分析", "nextActionPlan": ["3条具体可执行改进建议"] }`,
          '',
          `累计题数=${twin.totalSessions}，完题数=${twin.totalProblemsSolved}，主导短板=${WEAKNESS_LABELS[twin.habitProfile.dominantWeakness]}`,
          `习惯画像=${JSON.stringify(twin.habitProfile)}`,
          `薄弱知识点=${JSON.stringify(weakList)}`,
          latestTelemetry ? `最近一题过程摘要=${JSON.stringify({
            tags: latestTelemetry.knowledgeTags,
            progress: latestTelemetry.finalStepProgress,
            pitfalls: latestTelemetry.hitPitfalls,
            habits: latestTelemetry.habits,
          })}` : '',
        ]
          .filter(Boolean)
          .join('\n');

        const resp = await fetch(endpoint, {
          method: 'POST',
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${llmKey}`,
          },
          body: JSON.stringify({
            model: llmModel,
            stream: false,
            temperature: 0.25,
            max_tokens: 700,
            messages: [
              { role: 'system', content: '你只输出严格合法 JSON，不要附带 Markdown 代码块外多余说明。' },
              { role: 'user', content: prompt },
            ],
          }),
        });
        if (resp.ok) {
          const completion = (await resp.json()) as CompletionResponse;
          const rawContent = completion?.choices?.[0]?.message?.content;
          const content = typeof rawContent === 'string' ? rawContent : '';
          const parsed = parseCoachJson(content);
          if (parsed?.summary && parsed?.thinkingDiagnosis && parsed?.habitDiagnosis) {
            return {
              summary: String(parsed.summary),
              thinkingDiagnosis: String(parsed.thinkingDiagnosis),
              habitDiagnosis: String(parsed.habitDiagnosis),
              nextActionPlan: Array.isArray(parsed.nextActionPlan)
                ? parsed.nextActionPlan.map(String).slice(0, 4)
                : [],
              weakConcepts: weakList,
              dominantWeakness: twin.habitProfile.dominantWeakness,
              sourceMode: 'commercial_llm_edge',
              generatedAt: Date.now(),
            };
          }
        }
      } catch {
        // 回退至端侧本地启发诊断
      } finally {
        clearTimeout(timer);
      }
    }

    const hp = twin.habitProfile;
    const topWeak = weakList[0];
    const thinkingDiagnosis = topWeak
      ? `当前「${topWeak.label}」过程掌握度约 ${Math.round(topWeak.masteryProb * 100)}%（命中歧路 ${topWeak.pitfallHits} 次）；结构边界长停占比 ${Math.round(hp.structuralPauseRatio * 100)}%，建议在列式换元/分情况讨论前先写 1 行条件清单再动笔。`
      : `已在 ${twin.totalSessions} 次练习中建立初步轨迹，整体结构边界停顿占比 ${Math.round(hp.structuralPauseRatio * 100)}%。`;

    const habitDiagnosis =
      `平均书写动量 ${hp.avgMomentum}、单笔平均涂改/撤销率 ${hp.avgEraseRatio}、深度心算静音占比 ${Math.round(hp.deepThinkingRatio * 100)}%；当前主导改善方向为「${WEAKNESS_LABELS[hp.dominantWeakness]}」。`;

    const nextActionPlan: string[] = [];
    if (hp.avgEraseRatio >= 0.3) {
      nextActionPlan.push('涂改集中在代数变形段：建议草稿区先验算正负号再落主区。');
    }
    if (hp.structuralPauseRatio >= 0.55) {
      nextActionPlan.push('常在 = / ∫ / |A| 等大结构处长停：落笔前先在右上角列出可用定理名与已知边界。');
    }
    if (topWeak && topWeak.masteryProb < 0.55) {
      nextActionPlan.push(`未来 2 次专项加练「${topWeak.label}」，重点留意临界零点与积分/展开阶数校验。`);
    }
    if (nextActionPlan.length === 0) {
      nextActionPlan.push('保持当前心算与草稿分离节奏，尝试挑战综合压轴题并开启「苏格拉底」模式减少被动提示。');
    }

    return {
      summary: `已完成 ${twin.totalSessions} 题过程建模：${WEAKNESS_LABELS[hp.dominantWeakness]}。`,
      thinkingDiagnosis,
      habitDiagnosis,
      nextActionPlan,
      weakConcepts: weakList,
      dominantWeakness: hp.dominantWeakness,
      sourceMode: 'offline_heuristic',
      generatedAt: Date.now(),
    };
  }
}
