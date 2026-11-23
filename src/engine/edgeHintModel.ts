import {
  filterSafeHintText,
  levelInstructions,
  resolveChatEndpoint,
  styleInstructions,
  systemMessage,
  TIMEOUT_MS,
  type CompletionResponse,
  type HintLevel,
} from '../hints/safety';
import type { StallType } from '../stalls';
import type { HintStyle } from './personalizedPolicy';
import {
  getOfflineStyledHint,
  MATH_DOMAIN_KNOWLEDGE,
  type SolutionCacheEntry,
  type SolutionPitfallSpec,
} from './solutionCache';
import type { IEdgeHintModelAdapter } from './types';

export interface EdgeHintContext {
  solutionEntry?: SolutionCacheEntry | null;
  matchedStepIndex?: number;
  hitPitfall?: SolutionPitfallSpec | null;
  recognizedLatex?: string;
}

const EXTRA_MATH_CLUES: Array<{ pattern: RegExp; clue: string }> = [
  ...MATH_DOMAIN_KNOWLEDGE.map(({ pattern, clue }) => ({ pattern, clue })),
  { pattern: /单调|对称轴|抛物线|二次函数/, clue: '一阶导数正负符号变化或开口与边界条件' },
  { pattern: /ln\s*x|求导|极值|切线/, clue: '对数定义域 x > 0 与乘积/复合求导法则' },
  { pattern: /数列|通项|级数|收敛/, clue: '首项与通项验证或正项/交错级数判别法' },
];

/**
 * 端侧提示合成与本地小模型适配器（实现 IEdgeHintModelAdapter）。
 *
 * 【自研上下文合成 + 可选端侧本地大模型 HTTP 桥接】
 * 1. 若在本地配置 EXPO_PUBLIC_EDGE_LLM_URL（如真机 Termux llama.cpp / 局域网 Ollama 部署的 Qwen2.5-Math-1.5B），
 *    在断云网环境下调用本地推理接口（8 秒超时熔断 + 防剧透过滤）；
 * 2. 若无本地推理服务，则结合 SolutionCache 步骤图、易错歧路和当前手写表达式动态合成针对性离线提示。
 */
export class EdgeHintModelAdapter implements IEdgeHintModelAdapter {
  private contextProvider: (() => EdgeHintContext) | null = null;

  public bindContextProvider(provider: (() => EdgeHintContext) | null): void {
    this.contextProvider = provider;
  }

  public isReady(): boolean {
    const edgeUrl = process.env.EXPO_PUBLIC_EDGE_LLM_URL?.trim();
    if (edgeUrl) return true;
    const ctx = this.contextProvider?.();
    return Boolean(ctx?.solutionEntry || ctx?.hitPitfall || ctx?.recognizedLatex);
  }

  public async generateHint(params: {
    question: string;
    type: StallType;
    level: HintLevel;
    durationMs: number;
    style: HintStyle;
  }): Promise<string> {
    const ctx = this.contextProvider?.() ?? {};

    // 1. 若命中了明确的易错歧路特征，优先弹出靶向预警（毫秒级，零推理耗时）
    if (ctx.hitPitfall?.warningHint) {
      const safe = filterSafeHintText(ctx.hitPitfall.warningHint, ctx.solutionEntry?.sealedFinalAnswer);
      if (safe) return safe;
    }

    // 2. 若配置了本地局域网/真机端侧小模型端点（如 Ollama qwen2-math:1.5b / deepseek-r1:1.5b），尝试端侧推理
    const edgeBase = process.env.EXPO_PUBLIC_EDGE_LLM_URL?.trim().replace(/\/+$/, '');
    const edgeModel = process.env.EXPO_PUBLIC_EDGE_LLM_MODEL?.trim() || 'qwen2-math:1.5b';
    if (edgeBase && params.question.trim()) {
      const endpoint = resolveChatEndpoint(edgeBase, true);
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
      try {
        const stepNote =
          ctx.matchedStepIndex !== undefined &&
          ctx.matchedStepIndex >= 0 &&
          ctx.solutionEntry?.steps[ctx.matchedStepIndex]
            ? `当前疑卡在步骤${ctx.matchedStepIndex + 1}：${ctx.solutionEntry.steps[ctx.matchedStepIndex].goalDescription}`
            : '';
        const response = await fetch(endpoint, {
          method: 'POST',
          signal: controller.signal,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: edgeModel,
            stream: false,
            temperature: 0.2,
            max_tokens: 180,
            messages: [
              { role: 'system', content: systemMessage },
              {
                role: 'user',
                content: [
                  `题干：${params.question.trim()}`,
                  stepNote,
                  ctx.recognizedLatex ? `手写片段特征：${ctx.recognizedLatex}` : '',
                  `目标层级：L${params.level}（${levelInstructions[params.level]}）`,
                  styleInstructions[params.style],
                ]
                  .filter(Boolean)
                  .join('\n'),
              },
            ],
          }),
        });
        if (response.ok) {
          const data = (await response.json()) as CompletionResponse;
          const text = filterSafeHintText(
            data?.choices?.[0]?.message?.content,
            ctx.solutionEntry?.sealedFinalAnswer,
          );
          if (text) return text;
        }
      } catch {
        // 端侧本地推理超时或未启动时，无缝降级至步骤上下文离线合成
      } finally {
        clearTimeout(timeout);
      }
    }

    // 3. 纯 TypeScript 上下文合成（结合解题图谱步骤目标与题干关键词线索）
    return this.synthesizeLocalContextualHint(params, ctx);
  }

  private synthesizeLocalContextualHint(
    params: {
      question: string;
      type: StallType;
      level: HintLevel;
      style: HintStyle;
    },
    ctx: EdgeHintContext,
  ): string {
    const { solutionEntry, matchedStepIndex } = ctx;
    if (solutionEntry && solutionEntry.steps.length > 0) {
      const idx =
        matchedStepIndex !== undefined && matchedStepIndex >= 0
          ? Math.min(matchedStepIndex, solutionEntry.steps.length - 1)
          : 0;
      const step = solutionEntry.steps[idx];
      if (params.level === 0 && step?.goalDescription) {
        return `先自检第 ${idx + 1} 步「${step.goalDescription}」的符号与关键条件是否完整。`;
      }
      if (params.level === 1 && step?.goalDescription) {
        if (params.style === 'step_decompose') {
          return `先把当前目标拆小到这一步：尝试「${step.goalDescription}」。`;
        }
        if (params.style === 'sanity_check') {
          return `核对第 ${idx + 1} 步关键条件与特值是否匹配「${step.goalDescription}」。`;
        }
        if (params.style === 'concept_recall') {
          return `回顾一下完成「${step.goalDescription}」通常依赖哪条核心定理或公式？`;
        }
        return `当前推到第 ${idx + 1} 步附近了，试着围绕「${step.goalDescription}」找线索。`;
      }
      if (params.level === 2 && solutionEntry.steps[idx + 1]?.goalDescription) {
        return `完成当前过渡后，下一步可以尝试「${solutionEntry.steps[idx + 1].goalDescription}」。`;
      }
    }

    const q = params.question.trim();
    if (q) {
      const matchedClue = EXTRA_MATH_CLUES.find(c => c.pattern.test(q));
      if (matchedClue && params.level === 1) {
        return `结合这道题的核心条件，先检查一下「${matchedClue.clue}」有没有用上？`;
      }
    }

    return getOfflineStyledHint(params.type, params.level, params.style);
  }
}
