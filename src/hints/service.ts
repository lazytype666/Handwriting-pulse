import type { HintStyle } from '../engine/personalizedPolicy';
import { getOfflineStyledHint } from '../engine/solutionCache';
import type { StallType } from '../stalls';
import {
  levelInstructions,
  readHintFromCompletion,
  resolveChatEndpoint,
  styleInstructions,
  systemMessage,
  TIMEOUT_MS,
  type CompletionResponse,
  type HintLevel,
} from './safety';

export type { HintLevel } from './safety';

export interface HintRequest {
  question: string;
  type: StallType;
  level: HintLevel;
  durationMs: number;
  style?: HintStyle;
  recognizedLatex?: string;
  matchedStepIndex?: number;
  pitfallWarning?: string;
}

export interface HintResult {
  text: string;
  source: 'online' | 'offline' | 'cache';
  style?: HintStyle;
}

function offlineHint({ type, level, style }: HintRequest): HintResult {
  return {
    source: 'offline',
    style,
    text: getOfflineStyledHint(type, level, style ?? 'socratic_question'),
  };
}

// 平台自带的 fetch 直接调用 OpenAI 兼容服务；仅传题干和事件摘要，不传轨迹。
// Expo 在打包时只对 process.env.EXPO_PUBLIC_* 的静态点访问执行替换。
export async function getHint(request: HintRequest, signal?: AbortSignal): Promise<HintResult> {
  const fallback = offlineHint(request);
  const base = process.env.EXPO_PUBLIC_LLM_BASE_URL?.trim().replace(/\/+$/, '');
  const key = process.env.EXPO_PUBLIC_LLM_API_KEY?.trim();
  const model = process.env.EXPO_PUBLIC_LLM_MODEL?.trim();
  const question = request.question.trim();
  if (!question || !base || !key || !model || signal?.aborted) return fallback;
  const endpoint = resolveChatEndpoint(base, false);
  const controller = new AbortController();
  const abort = () => controller.abort();
  const deadline = new Promise<HintResult>(resolve => {
    controller.signal.addEventListener('abort', () => resolve(fallback), { once: true });
  });
  signal?.addEventListener('abort', abort);
  const timeout = setTimeout(abort, TIMEOUT_MS);
  try {
    const triggerText =
      request.type === 'pause_erase'
        ? '停顿涂改'
        : request.type === 'speed_drop'
          ? '速度骤降'
          : '长时阻滞';
    // 将响应体读取也纳入八秒时限。部分 fetch 实现在响应头到达后仍可能挂起。
    const completion = async (): Promise<HintResult> => {
      const response = await fetch(endpoint, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + key },
        body: JSON.stringify({
          model,
          stream: false,
          messages: [
            { role: 'system', content: systemMessage },
            {
              role: 'user',
              content: JSON.stringify({
                题干: question,
                触发类型: triggerText,
                停顿时长毫秒: request.type === 'pause_erase' ? request.durationMs : 0,
                减速持续毫秒: request.type === 'speed_drop' ? request.durationMs : 0,
                目标层级: 'L' + request.level,
                提示要求: levelInstructions[request.level],
                ...(request.style ? { 风格要求: styleInstructions[request.style] } : {}),
                ...(request.recognizedLatex ? { 局部识别算式: request.recognizedLatex } : {}),
                ...(request.matchedStepIndex !== undefined && request.matchedStepIndex >= 0
                  ? { 疑卡步骤序号: request.matchedStepIndex + 1 }
                  : {}),
              }),
            },
          ],
        }),
      });
      if (!response.ok) return fallback;
      const hint = readHintFromCompletion((await response.json()) as CompletionResponse);
      return hint ? { source: 'online', style: request.style, text: hint } : fallback;
    };
    return await Promise.race([completion(), deadline]);
  } catch {
    return fallback;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener('abort', abort);
  }
}
