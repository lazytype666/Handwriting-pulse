import * as FileSystem from 'expo-file-system/legacy';
import { TIMEOUT_MS, type CompletionResponse } from '../hints/safety';
import { SAMPLE_PROBLEMS, type SampleProblemSpec } from './sampleProblems';
import { normalizeProblemKey } from './solutionCache';
import type { IProblemIngestorAdapter } from './types';

export interface ProblemContentBlock {
  type: 'text' | 'inline_formula' | 'display_formula';
  content: string;
}

export interface IngestedProblemResult {
  problemId: string;
  problemText: string;
  blocks: ProblemContentBlock[];
  matchedSample: SampleProblemSpec | null;
  confidence: number;
  needsUserConfirm: boolean;
}

const VLM_OCR_SYSTEM_PROMPT = [
  '你是数学试卷排版与公式识别助手。只提取图片中的数学题目文字与公式，输出规范的 Markdown + LaTeX 题干，不解题、不给出任何解答步骤或答案。',
  '行内公式用 $...$ 包裹，独立公式块用 $$...$$ 包裹，确保条件、区间和变量字母完全准确。',
].join('');

function hashProblemText(text: string): string {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return 'prob_' + (h >>> 0).toString(16).padStart(8, '0');
}

/** 将含 $...$ / $$...$$ 的数学题干切分为结构化文本块与公式块（用于版面语义解析） */
export function parseLatexBlocks(rawText: string): ProblemContentBlock[] {
  const blocks: ProblemContentBlock[] = [];
  const regex = /(\$\$[\s\S]+?\$\$|\$[^$\n]+\$)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(rawText)) !== null) {
    if (match.index > lastIndex) {
      const plain = rawText.slice(lastIndex, match.index).trim();
      if (plain) blocks.push({ type: 'text', content: plain });
    }
    const token = match[0];
    if (token.startsWith('$$') && token.endsWith('$$')) {
      blocks.push({ type: 'display_formula', content: token.slice(2, -2).trim() });
    } else {
      blocks.push({ type: 'inline_formula', content: token.slice(1, -1).trim() });
    }
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < rawText.length) {
    const tail = rawText.slice(lastIndex).trim();
    if (tail) blocks.push({ type: 'text', content: tail });
  }
  return blocks;
}

/**
 * 题目多模态/结构化摄入适配器（实现 IProblemIngestorAdapter）。
 *
 * 【自研实现 + 兼容外部 VLM 服务】
 * - 文本摄入：端侧完成 LaTeX 块结构拆解与离线示例题库对齐；
 * - 图像摄入：在真机配置 VLM 视觉模型环境变量时，通过内置 fetch 调用兼容服务提取题干文字与 LaTeX。
 */
export class ProblemIngestorAdapter implements IProblemIngestorAdapter {
  public async ingestFromText(rawText: string): Promise<IngestedProblemResult> {
    const cleaned = normalizeProblemKey(rawText);
    const problemId = hashProblemText(cleaned);
    const blocks = parseLatexBlocks(cleaned);
    const matchedSample =
      SAMPLE_PROBLEMS.find(s => normalizeProblemKey(s.problemText) === cleaned) ?? null;

    return {
      problemId,
      problemText: cleaned,
      blocks,
      matchedSample,
      confidence: 1.0,
      needsUserConfirm: false,
    };
  }

  public async ingestFromImageUri(imageUri: string): Promise<IngestedProblemResult> {
    const base = (
      process.env.EXPO_PUBLIC_VLM_BASE_URL ?? process.env.EXPO_PUBLIC_LLM_BASE_URL
    )
      ?.trim()
      .replace(/\/+$/, '');
    const apiKey = (
      process.env.EXPO_PUBLIC_VLM_API_KEY ?? process.env.EXPO_PUBLIC_LLM_API_KEY
    )?.trim();
    const model = process.env.EXPO_PUBLIC_VLM_MODEL?.trim();

    if (!base || !apiKey || !model) {
      throw new Error('未配置视觉识题环境变量（EXPO_PUBLIC_VLM_MODEL），请手动粘贴题干或使用预设示例。');
    }

    let dataUrl = imageUri.trim();
    if (!dataUrl.startsWith('data:')) {
      const base64 = await FileSystem.readAsStringAsync(dataUrl, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const ext = dataUrl.toLowerCase().endsWith('.png') ? 'png' : 'jpeg';
      dataUrl = `data:image/${ext};base64,${base64}`;
    }

    const endpoint = base.endsWith('/chat/completions') ? base : base + '/chat/completions';
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + apiKey },
        body: JSON.stringify({
          model,
          stream: false,
          messages: [
            { role: 'system', content: VLM_OCR_SYSTEM_PROMPT },
            {
              role: 'user',
              content: [
                { type: 'text', text: '请识别图中数学题目的完整题干（只提取题目本身，不要解答）。' },
                { type: 'image_url', image_url: { url: dataUrl } },
              ],
            },
          ],
        }),
      });
      if (!response.ok) {
        throw new Error(`题目图像识别服务返回状态 ${response.status}`);
      }
      const completion = (await response.json()) as CompletionResponse;
      const content = completion?.choices?.[0]?.message?.content;
      if (typeof content !== 'string' || !content.trim()) {
        throw new Error('视觉模型未返回有效题干文本');
      }
      const result = await this.ingestFromText(content);
      return {
        ...result,
        confidence: 0.9,
        needsUserConfirm: true,
      };
    } finally {
      clearTimeout(timeout);
    }
  }
}
