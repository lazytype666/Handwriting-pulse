import { useCallback, useEffect, useRef, useState } from 'react';
import { getHint, type HintLevel, type HintRequest } from './service';

type TriggerEvent = Pick<HintRequest, 'type' | 'durationMs'>;
interface CurrentHint {
  level: HintLevel;
  type: HintRequest['type'];
  source: 'loading' | 'online' | 'offline';
  text: string;
}

const DISPLAY_MS = 8000;

export function useHints(question: string) {
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
  }, [clear]);

  // 换题、关闭或换层时撤销旧请求；迟到的网络结果不能覆盖另一层提示。
  useEffect(() => {
    dismiss();
    return clear;
  }, [question, clear, dismiss]);

  const show = useCallback((request: HintRequest) => {
    clear();
    const controller = new AbortController();
    active.current = { request, controller };
    const { type, level } = request;
    setCurrent({ level, type, source: 'loading', text: '正在准备提示…' });
    void getHint(request, controller.signal).then(result => {
      if (active.current?.controller !== controller) return;
      setCurrent({ level, type, ...result });
      // 从提示抵达开始计时，让慢网络下的每一层也有完整的阅读时间。
      timer.current = setTimeout(dismiss, DISPLAY_MS);
    });
  }, [clear, dismiss]);

  const trigger = useCallback(({ type, durationMs }: TriggerEvent) => {
    show({ question, type, durationMs, level: 1 });
  }, [question, show]);
  const advance = useCallback(() => {
    const previous = active.current?.request;
    if (previous && previous.level < 3) show({ ...previous, level: previous.level === 1 ? 2 : 3 });
  }, [show]);
  return { current, trigger, advance, dismiss };
}
