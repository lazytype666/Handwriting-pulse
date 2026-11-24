import * as FileSystem from 'expo-file-system/legacy';
import {
  isLocalModelUrl,
  resolveChatEndpoint,
  type CompletionResponse,
} from '../hints/safety';
import { parseMathSegments } from '../utils/mathFormat';
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
  ocrSource?: 'text' | 'mlkit_local' | 'mlkit_llm_refined' | 'vlm_local' | 'vlm_cloud';
}

const COMMERCIAL_VLM_TIMEOUT_MS = 15000;
const LOCAL_VLM_TIMEOUT_MS = 45000;

const VLM_OCR_SYSTEM_PROMPT = [
  '你是大学数学试卷排版与公式识别助手。只提取图片中的数学题目完整文字与公式，输出规范的 Markdown + LaTeX 题干，绝不解题、不给出任何解答步骤或答案。',
  '行内公式用 $...$ 包裹，独立公式块用 $$...$$ 包裹；特别注意精准还原分式 \\frac{}{}、多重/曲线积分 \\iint/\\oint、极限下标 \\lim_{x \\to 0}、分段函数 \\begin{cases}...\\end{cases}、矩阵/行列式 \\begin{bmatrix}/\\begin{vmatrix}、偏导 \\partial、梯度 \\nabla 与希腊字母。',
].join('');

const OCR_REPAIR_SYSTEM_PROMPT = [
  '你是数学 OCR 文本规范化助手。输入是从手机端 OCR 提取的混排纯文本（可能把分式、上下标或积分上下限压扁了）。',
  '请完整保留原有的中文/英文题干措辞与标点，只把其中的数学表达式改写为标准 $...$ LaTeX 格式；仅返回规范化后的整句题干，绝不解题、不化简计算、不输出答案。',
].join('');

function hashProblemText(text: string): string {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return 'prob_' + (h >>> 0).toString(16).padStart(8, '0');
}

/** 统一解析视觉识题端点：支持单独 VLM 配置或与商用解题大模型共用 Base URL / API Key */
export function resolveVlmConnection(): {
  base: string;
  apiKey?: string;
  model: string;
  isLocal: boolean;
} | null {
  const base = (
    process.env.EXPO_PUBLIC_VLM_BASE_URL ??
    process.env.EXPO_PUBLIC_LLM_BASE_URL ??
    process.env.EXPO_PUBLIC_EDGE_LLM_URL
  )
    ?.trim()
    .replace(/\/+$/, '');
  const apiKey = (
    process.env.EXPO_PUBLIC_VLM_API_KEY ?? process.env.EXPO_PUBLIC_LLM_API_KEY
  )?.trim();
  const explicitVlmModel = process.env.EXPO_PUBLIC_VLM_MODEL?.trim();
  const sharedLlmModel = process.env.EXPO_PUBLIC_LLM_MODEL?.trim();
  const model =
    explicitVlmModel ||
    (sharedLlmModel && /vl|vision|gemini|gpt-4|gpt-5|claude|qvq|multimodal/i.test(sharedLlmModel)
      ? sharedLlmModel
      : '');
  if (!base || !model) return null;
  const isLocal = isLocalModelUrl(base);
  if (!isLocal && !apiKey) return null;
  return { base, apiKey: apiKey || undefined, model, isLocal };
}

/** 对设备端 ML Kit OCR 的常见数学扁平化误差做零耗时启发式规整（仅无 VLM 纯离线分支使用） */
export function heuristicNormalizeOcrMathText(rawText: string): string {
  return rawText
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/一>/g, '->')
    .replace(/—>/g, '->')
    .replace(/([A-Za-z0-9)\]])\s*²/g, '$1^2')
    .replace(/([A-Za-z0-9)\]])\s*³/g, '$1^3')
    .replace(/([A-Za-z0-9)\]])\s*⁻¹/g, '$1^{-1}')
    .trim();
}

/** 将含 $...$ / $$...$$ / \(...\) / \[...\] 的数学题干切分为结构化文本块与公式块（用于版面语义解析与前端预览） */
export function parseLatexBlocks(rawText: string): ProblemContentBlock[] {
  return parseMathSegments(rawText)
    .map((seg) => ({
      type:
        seg.type === 'display_math'
          ? ('display_formula' as const)
          : seg.type === 'inline_math'
            ? ('inline_formula' as const)
            : ('text' as const),
      content: seg.raw.trim(),
    }))
    .filter((b) => b.content.length > 0);
}

/**
 * 题目多模态/结构化摄入适配器（实现 IProblemIngestorAdapter）。
 *
 * 【升级版商用 VLM 优先策略】
 * 1. 图像摄入：支持与解题大模型共用同一商用 API（同一 Base URL + API Key），15s 云端超时 + 自动瞬断重试；
 *    不再混入乱码 ML Kit hint，失败时保留裁剪图供一键重识而非污染输入框；
 * 2. 文本摄入：端侧完成 LaTeX 块结构拆解与离线示例题库对齐。
 */
export class ProblemIngestorAdapter implements IProblemIngestorAdapter {
  public async ingestFromText(
    rawText: string,
    ocrSource: IngestedProblemResult['ocrSource'] = 'text',
    needsUserConfirm = false,
  ): Promise<IngestedProblemResult> {
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
      confidence: ocrSource === 'text' ? 1.0 : 0.88,
      needsUserConfirm,
      ocrSource,
    };
  }

  /**
   * 对纯离线 `rn-mlkit-ocr` 文本（无 VLM 端点配置时）进行数学格式归一化。
   */
  public async refineOcrText(rawOcrText: string): Promise<IngestedProblemResult> {
    const normalized = heuristicNormalizeOcrMathText(rawOcrText);
    if (!normalized) {
      return this.ingestFromText('', 'mlkit_local', true);
    }

    const vlmConn = resolveVlmConnection();
    const edgeBase = process.env.EXPO_PUBLIC_EDGE_LLM_URL?.trim();
    const edgeModel = process.env.EXPO_PUBLIC_EDGE_LLM_MODEL?.trim() || 'qwen2-math:1.5b';
    const cloudBase = process.env.EXPO_PUBLIC_LLM_BASE_URL?.trim();
    const cloudKey = process.env.EXPO_PUBLIC_LLM_API_KEY?.trim();
    const cloudModel = process.env.EXPO_PUBLIC_LLM_MODEL?.trim();

    const useEdge = !vlmConn && Boolean(edgeBase);
    const useCloud = !vlmConn && !useEdge && Boolean(cloudBase && cloudKey && cloudModel);
    const looksMath = /[=+\-*/^_{}()|∫∑√λΦ]|lim|sin|cos|ln|det|矩阵|积分|概率/i.test(normalized);

    if ((vlmConn || useEdge || useCloud) && looksMath) {
      const targetBase = vlmConn ? vlmConn.base : useEdge ? edgeBase! : cloudBase!;
      const localMode = vlmConn ? vlmConn.isLocal : useEdge;
      const endpoint = resolveChatEndpoint(targetBase, localMode);
      const model = vlmConn ? vlmConn.model : useEdge ? edgeModel : cloudModel!;
      const authKey = vlmConn ? vlmConn.apiKey : useCloud ? cloudKey : undefined;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), localMode ? 10000 : 8000);
      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            ...(authKey ? { Authorization: 'Bearer ' + authKey } : {}),
          },
          body: JSON.stringify({
            model,
            stream: false,
            temperature: 0.05,
            max_tokens: 400,
            messages: [
              { role: 'system', content: OCR_REPAIR_SYSTEM_PROMPT },
              {
                role: 'user',
                content:
                  '示例输入：求函数 y = x2 的导数。\n示例输出：求函数 $y = x^2$ 的导数。\n\n请转写：' +
                  normalized,
              },
            ],
          }),
        });
        if (res.ok) {
          const data = (await res.json()) as CompletionResponse;
          const content = data?.choices?.[0]?.message?.content;
          if (typeof content === 'string') {
            const repaired = content
              .replace(/<think>[\s\S]*?<\/think>/gi, '')
              .replace(/<think>[\s\S]*$/gi, '')
              .replace(/^```(?:markdown|latex|md)?\s*/i, '')
              .replace(/```$/i, '')
              .replace(/^(?:输出|示例输出)[:：]\s*/i, '')
              .trim();
            if (
              repaired &&
              repaired.length <= Math.max(240, normalized.length * 2.4) &&
              !/\\boxed|解[:：]|答案[:：]|To find the/i.test(repaired)
            ) {
              return this.ingestFromText(repaired, 'mlkit_llm_refined', true);
            }
          }
        }
      } catch {
        // 离线兜底
      } finally {
        clearTimeout(timeout);
      }
    }

    return this.ingestFromText(normalized, 'mlkit_local', true);
  }

  public async ingestFromImageUri(
    imageUri: string,
    hintMlkitText?: string,
  ): Promise<IngestedProblemResult> {
    const conn = resolveVlmConnection();
    if (!conn) {
      if (hintMlkitText?.trim()) {
        return this.refineOcrText(hintMlkitText);
      }
      throw new Error(
        '未配置视觉识题 API（请在 .env.local 配置 EXPO_PUBLIC_VLM_MODEL 与商用 LLM Base URL / API Key）。',
      );
    }

    let dataUrl = imageUri.trim();
    if (!dataUrl.startsWith('data:')) {
      const base64 = await FileSystem.readAsStringAsync(dataUrl, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const ext = dataUrl.toLowerCase().endsWith('.png') ? 'png' : 'jpeg';
      dataUrl = `data:image/${ext};base64,${base64}`;
    }

    const endpoint = resolveChatEndpoint(conn.base, conn.isLocal);
    const timeoutMs = conn.isLocal ? LOCAL_VLM_TIMEOUT_MS : COMMERCIAL_VLM_TIMEOUT_MS;

    const requestVlmOnce = async (): Promise<IngestedProblemResult> => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            ...(conn.apiKey ? { Authorization: 'Bearer ' + conn.apiKey } : {}),
          },
          body: JSON.stringify({
            model: conn.model,
            stream: false,
            temperature: 0.1,
            max_tokens: 800,
            messages: [
              { role: 'system', content: VLM_OCR_SYSTEM_PROMPT },
              {
                role: 'user',
                content: [
                  {
                    type: 'text',
                    text: '请精确识别图中数学题目的完整题干，输出 Markdown + LaTeX（公式用 $...$ 或 $$...$$ 包裹），不要解题或给提示。',
                  },
                  { type: 'image_url', image_url: { url: dataUrl } },
                ],
              },
            ],
          }),
        });
        if (!response.ok) {
          throw new Error(`视觉识题 API 返回状态 ${response.status}`);
        }
        const completion = (await response.json()) as CompletionResponse;
        const rawContent = completion?.choices?.[0]?.message?.content;
        if (typeof rawContent !== 'string' || !rawContent.trim()) {
          throw new Error('视觉模型未返回有效题干文本');
        }
        const cleaned = rawContent
          .replace(/<think>[\s\S]*?<\/think>/gi, '')
          .replace(/^```(?:markdown|latex|md)?\s*/i, '')
          .replace(/```$/i, '')
          .trim();
        const result = await this.ingestFromText(
          cleaned,
          conn.isLocal ? 'vlm_local' : 'vlm_cloud',
          true,
        );
        return {
          ...result,
          confidence: conn.isLocal ? 0.92 : 0.96,
          needsUserConfirm: true,
        };
      } finally {
        clearTimeout(timeout);
      }
    };

    try {
      try {
        return await requestVlmOnce();
      } catch (firstErr) {
        const msg = firstErr instanceof Error ? firstErr.message : String(firstErr);
        const transient = /canceled|aborted|network request failed|timed out|429|502|503|504/i.test(msg);
        if (transient) {
          return await requestVlmOnce();
        }
        throw firstErr;
      }
    } catch (err) {
      // 商用 VLM 失败时不往输入框强塞乱码，而是报错并允许前端一键用已裁剪缓存图重发
      const msg = err instanceof Error ? err.message : String(err);
      if (/canceled|aborted|network request failed|timed out/i.test(msg)) {
        throw new Error(
          conn.isLocal
            ? '本地视觉识题超时或连接中断，请确认 Ollama 服务及热点连通后点「用原裁剪图重识」。'
            : '云端视觉识题请求超时或网络抖动（已自动重试一次），请点「用原裁剪图重识」或直接粘贴题目。',
        );
      }
      throw err;
    }
  }
}
