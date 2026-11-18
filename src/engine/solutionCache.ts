import {
  filterSafeHintText,
  systemMessage,
  TIMEOUT_MS,
  type CompletionResponse,
  type HintLevel,
} from '../hints/safety';
import type { StallType } from '../stalls';
import {
  HINT_STYLES,
  HINT_STYLE_LABELS,
  type HintStyle,
} from './personalizedPolicy';

export interface SolutionStepSpec {
  stepIndex: number;
  goalDescription: string;
  expectedTokens: string[];
}

export interface SolutionPitfallSpec {
  triggerPattern: string;
  warningHint: string;
}

export interface StyleHintTiers {
  l1?: string;
  l2?: string;
  l3?: string;
}

export interface SolutionCacheEntry {
  problemKey: string;
  problemText: string;
  steps: SolutionStepSpec[];
  commonPitfalls: SolutionPitfallSpec[];
  hintsByStyle: Partial<Record<HintStyle, StyleHintTiers>>;
  /** 密封的最终答案占位（仅预留给后续判对错模型使用，严禁展示给学生） */
  sealedFinalAnswer?: string;
  sourceMode: 'cloud_prewarm' | 'cloud_live' | 'edge_offline';
  updatedAt: number;
}

export function normalizeProblemKey(text: string): string {
  return text.trim().replace(/\s+/g, ' ');
}

/**
 * 本地离线多风格分层提示库（无网/无题干/超时降级时保证 4 种风格全部可用，且默认反问风格保留阶段二原文）。
 */
export function getOfflineStyledHint(
  type: StallType,
  level: HintLevel,
  style: HintStyle = 'socratic_question',
): string {
  if (style === 'socratic_question') {
    if (level === 3) return '还是没思路的话，先标记这道题，休息几分钟再回来看。';
    if (level === 2) return '试着把当前这一步拆成更小的目标，一步一步来。';
    if (type === 'pause_erase') {
      return '这里卡住了吗？试试把题目的条件再读一遍，看看有没有没用上的。';
    }
    if (type === 'speed_drop') {
      return '思路断了吗？回想一下上一步用到了什么结论。';
    }
    return '思考遇阻了吗？先问自己：这一步最终想消掉或求出哪个关键量？';
  }

  if (style === 'condition_reminder') {
    if (level === 1) return '先停笔扫一遍题干，看看哪个已知条件或取值范围还没用上？';
    if (level === 2) return '把题干中的已知关系逐条列出来，检查还缺哪一环即可衔接。';
    return '先把用到的定理前提和题目给定条件逐一对齐，再确定下一步消元方向。';
  }

  if (style === 'algebra_check') {
    if (level === 1) return '刚才那一步变号或移项有没有小疏漏？回头核对一下系数与符号。';
    if (level === 2) return '试着代一个简单特值检验当前算式，确认等价变形没有算偏。';
    return '检查高次项系数是否为零、分母与定义域边界是否遗漏讨论，确认无误再往下推。';
  }

  if (level === 1) return '这一步跨度有些大，能不能先在中间插一个更小的过渡目标？';
  if (level === 2) return '试着把当前这一步拆成更小的目标，先写最确定的一半。';
  return '把大问题拆成“先求中间量 → 再代回原式”两小段，先只完成第一段推导。';
}

/**
 * 解题图谱缓存器（SolutionCache）：在学生审题期间静默预解步骤图与 4 风格三级提示，卡壳时毫秒级命中。
 */
export class SolutionCacheManager {
  private entries = new Map<string, SolutionCacheEntry>();
  private inflightPrewarm = new Map<string, Promise<SolutionCacheEntry | null>>();

  public hasPrewarmed(problemText: string): boolean {
    const key = normalizeProblemKey(problemText);
    if (!key) return false;
    const entry = this.entries.get(key);
    return Boolean(entry && Object.keys(entry.hintsByStyle).length > 0);
  }

  public lookupHint(
    problemText: string,
    level: HintLevel,
    style: HintStyle,
  ): { text: string; sourceMode: SolutionCacheEntry['sourceMode'] } | null {
    const key = normalizeProblemKey(problemText);
    if (!key) return null;
    const entry = this.entries.get(key);
    if (!entry) return null;

    const tiers = entry.hintsByStyle[style] ?? entry.hintsByStyle.socratic_question;
    if (!tiers) return null;
    const text = level === 1 ? tiers.l1 : level === 2 ? tiers.l2 : tiers.l3;
    if (!text) return null;
    return { text, sourceMode: entry.sourceMode };
  }

  public storeHint(
    problemText: string,
    level: HintLevel,
    style: HintStyle,
    hintText: string,
    sourceMode: SolutionCacheEntry['sourceMode'] = 'cloud_live',
  ): void {
    const key = normalizeProblemKey(problemText);
    if (!key) return;
    const existing = this.entries.get(key) ?? {
      problemKey: key,
      problemText: problemText.trim(),
      steps: [],
      commonPitfalls: [],
      hintsByStyle: {},
      sourceMode,
      updatedAt: Date.now(),
    };
    const tiers = existing.hintsByStyle[style] ?? {};
    if (level === 1) tiers.l1 = hintText;
    else if (level === 2) tiers.l2 = hintText;
    else tiers.l3 = hintText;
    existing.hintsByStyle[style] = tiers;
    existing.updatedAt = Date.now();
    this.entries.set(key, existing);
  }

  public async prewarm(problemText: string): Promise<SolutionCacheEntry | null> {
    const key = normalizeProblemKey(problemText);
    const base = process.env.EXPO_PUBLIC_LLM_BASE_URL?.trim().replace(/\/+$/, '');
    const apiKey = process.env.EXPO_PUBLIC_LLM_API_KEY?.trim();
    const model = process.env.EXPO_PUBLIC_LLM_MODEL?.trim();
    if (!key || !base || !apiKey || !model) return null;
    if (this.hasPrewarmed(key)) return this.entries.get(key) ?? null;

    const inflight = this.inflightPrewarm.get(key);
    if (inflight) return inflight;

    const endpoint = base.endsWith('/chat/completions') ? base : base + '/chat/completions';
    const task = (async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
      try {
        const styleList = HINT_STYLES.map(s => `${s}(${HINT_STYLE_LABELS[s]})`).join('、');
        const userPrompt = [
          `题干：${problemText.trim()}`,
          `请为该题生成结构化解题图谱 JSON（仅返回纯 JSON，不要代码块标记）：包含主干步骤 steps、易错歧路 commonPitfalls、以及 ${styleList} 四种风格的 l1/l2/l3 提示。`,
          '每条 l1/l2/l3 严格遵守老师提示规则：引用题干关键词，用普通文本表达数学式，绝不出现最终答案或完整过程。',
          JSON.stringify({
            steps: [{ stepIndex: 1, goalDescription: '步骤目标', expectedTokens: ['关键表达式'] }],
            commonPitfalls: [{ triggerPattern: '典型误区', warningHint: '温和反问' }],
            hintsByStyle: {
              socratic_question: { l1: '...', l2: '...', l3: '...' },
              condition_reminder: { l1: '...', l2: '...', l3: '...' },
              algebra_check: { l1: '...', l2: '...', l3: '...' },
              step_decompose: { l1: '...', l2: '...', l3: '...' },
            },
          }),
        ].join('\n\n');

        const response = await fetch(endpoint, {
          method: 'POST',
          signal: controller.signal,
          headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + apiKey },
          body: JSON.stringify({
            model,
            stream: false,
            messages: [
              { role: 'system', content: systemMessage },
              { role: 'user', content: userPrompt },
            ],
          }),
        });
        if (!response.ok) return null;

        const completion = (await response.json()) as CompletionResponse;
        const rawContent = completion?.choices?.[0]?.message?.content;
        if (typeof rawContent !== 'string') return null;

        const cleaned = rawContent.trim().replace(/^```(?:json)?/i, '').replace(/```$/i, '').trim();
        const parsed = JSON.parse(cleaned) as Partial<SolutionCacheEntry>;
        const sanitizedStyles: Partial<Record<HintStyle, StyleHintTiers>> = {};

        for (const style of HINT_STYLES) {
          const tiers = parsed.hintsByStyle?.[style];
          if (!tiers) continue;
          const l1 = filterSafeHintText(tiers.l1);
          const l2 = filterSafeHintText(tiers.l2);
          const l3 = filterSafeHintText(tiers.l3);
          if (l1 || l2 || l3) {
            sanitizedStyles[style] = {
              ...(l1 ? { l1 } : {}),
              ...(l2 ? { l2 } : {}),
              ...(l3 ? { l3 } : {}),
            };
          }
        }

        if (Object.keys(sanitizedStyles).length === 0) return null;
        const entry: SolutionCacheEntry = {
          problemKey: key,
          problemText: problemText.trim(),
          steps: Array.isArray(parsed.steps) ? parsed.steps : [],
          commonPitfalls: Array.isArray(parsed.commonPitfalls) ? parsed.commonPitfalls : [],
          hintsByStyle: sanitizedStyles,
          sealedFinalAnswer: parsed.sealedFinalAnswer,
          sourceMode: 'cloud_prewarm',
          updatedAt: Date.now(),
        };
        this.entries.set(key, entry);
        return entry;
      } catch {
        return null;
      } finally {
        clearTimeout(timeout);
        this.inflightPrewarm.delete(key);
      }
    })();

    this.inflightPrewarm.set(key, task);
    return task;
  }

  public exportAllEntries(): SolutionCacheEntry[] {
    return Array.from(this.entries.values());
  }
}
