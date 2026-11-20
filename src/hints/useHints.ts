import { useCallback, useEffect, useRef, useState } from 'react';
import type { HandwritingPulseEngine, HintStyle } from '../engine';
import { getHint, type HintLevel, type HintRequest } from './service';

type TriggerEvent = Pick<HintRequest, 'type' | 'durationMs'>;

export interface CurrentHint {
  level: HintLevel;
  type: HintRequest['type'];
  source: 'loading' | 'online' | 'offline' | 'cache';
  style?: HintStyle;
  text: string;
}

const DISPLAY_MS = 8000;

export function useHints(
  question: string,
  engine?: HandwritingPulseEngine,
  onFeedbackRecorded?: () => void,
) {
  const [current, setCurrent] = useState<CurrentHint | null>(null);
  const active = useRef<{ request: HintRequest; controller: AbortController } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const clear = useCallback(() => {
    active.current?.controller.abort();
    active.current = null;
    clearTimeout(timer.current);
  }, []);

  const dismiss = useCallback(() => {
    clear();
    setCurrent(null);
    if (engine) {
      const entry = engine.recordExplicitFeedback('dismiss');
      if (entry) onFeedbackRecorded?.();
    }
  }, [clear, engine, onFeedbackRecorded]);

  const dismissWithThinking = useCallback(() => {
    clear();
    setCurrent(null);
    if (engine) {
      const entry = engine.recordExplicitFeedback('thinking');
      if (entry) onFeedbackRecorded?.();
    }
  }, [clear, engine, onFeedbackRecorded]);

  // 换题时清空当前提示但不算作用户负反馈
  useEffect(() => {
    clear();
    setCurrent(null);
    return clear;
  }, [question, clear]);

  const show = useCallback(
    (request: HintRequest) => {
      clear();
      const controller = new AbortController();
      const style = engine ? engine.selectStyleForStall(request.type, request.style) : request.style;
      const syntaxCtx = engine?.getSyntaxContextForHint() ?? {};
      const enriched: HintRequest = { ...request, ...syntaxCtx, style };
      active.current = { request: enriched, controller };
      const { type, level } = enriched;

      const scheduleAutoHide = (resolvedStyle?: HintStyle) => {
        if (engine && resolvedStyle) engine.beginHintObservation(resolvedStyle);
        timer.current = setTimeout(() => {
          clear();
          setCurrent(null);
        }, DISPLAY_MS);
      };

      // 1. 若解题图谱缓存已命中，毫秒级直接弹出
      if (engine && style && enriched.question.trim()) {
        const cached = engine.lookupCachedHint(enriched.question, level, style);
        if (cached) {
          setCurrent({ level, type, source: 'cache', style, text: cached });
          scheduleAutoHide(style);
          return;
        }
      }

      setCurrent({ level, type, source: 'loading', style, text: '正在准备提示…' });
      void getHint(enriched, controller.signal).then(async result => {
        if (active.current?.controller !== controller) return;
        let finalResult = result;

        // 2. 若在线请求降级而端侧推理适配器就绪，则尝试由端侧模型兜底生成
        if (finalResult.source === 'offline' && engine && style && enriched.question.trim()) {
          const edgeText = await engine.tryEdgeAdapterHint({
            question: enriched.question,
            type,
            level,
            durationMs: enriched.durationMs,
            style,
          });
          if (active.current?.controller !== controller) return;
          if (edgeText) {
            finalResult = { source: 'offline', style, text: edgeText };
          }
        } else if (finalResult.source === 'online' && engine && style && enriched.question.trim()) {
          engine.cacheOnlineHint(enriched.question, level, style, finalResult.text);
        }

        setCurrent({ level, type, ...finalResult });
        scheduleAutoHide(finalResult.style ?? style);
      });
    },
    [clear, engine],
  );

  const trigger = useCallback(
    ({ type, durationMs }: TriggerEvent, preferredStyle?: HintStyle, initialLevel: HintLevel = 1) => {
      show({ question, type, durationMs, level: initialLevel, style: preferredStyle });
    },
    [question, show],
  );

  const advance = useCallback(() => {
    const previous = active.current?.request;
    if (previous && previous.level < 3) {
      if (engine) {
        const entry = engine.recordExplicitFeedback('request_more');
        if (entry) onFeedbackRecorded?.();
      }
      const nextLevel: HintLevel =
        previous.level === 0 ? 1 : previous.level === 1 ? 2 : 3;
      show({ ...previous, level: nextLevel });
    }
  }, [engine, onFeedbackRecorded, show]);

  return { current, trigger, advance, dismiss, dismissWithThinking };
}
