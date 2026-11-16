import { forwardRef, memo, useCallback, useImperativeHandle, useState } from 'react';
import { StyleSheet } from 'react-native';
import { Canvas, Path, Skia } from '@shopify/react-native-skia';
import {
  GestureDetector,
  GestureStateManager,
  useManualGesture,
} from 'react-native-gesture-handler';
import { useFrameCallback, useSharedValue } from 'react-native-reanimated';
import { runOnUIAsync, scheduleOnRN } from 'react-native-worklets';

import {
  appendPoint, beginStroke, copyStrokes, createRecorder, readDiagnostics,
  type Diagnostics, type Stroke,
} from './recorder';

export interface InkCanvasHandle {
  snapshot(): Promise<Stroke[]>;
}

interface InkCanvasProps {
  onDiagnostics: (diagnostics: Diagnostics) => void;
}

export const InkCanvas = memo(forwardRef<InkCanvasHandle, InkCanvasProps>(
  function InkCanvas({ onDiagnostics }, ref) {
    // 这份记录只由 UI worklet 持有和更新。JS 在导出屏障处取得独立副本，
    // 单点不触发 React 状态更新，也不在移动时跨线程复制全页数组。
    const recorder = useSharedValue(createRecorder());
    useImperativeHandle(ref, () => ({
      snapshot: () => runOnUIAsync(() => {
        'worklet';
        return copyStrokes(recorder.value);
      }),
    }), [recorder]);

    // Skia 的不可变路径用于显示，PathBuilder 用于累计轨迹。
    // 书写中仅刷新当前笔的快照，完成笔只在收笔时并入底层路径。
    const [geometry] = useState(() => ({
      live: Skia.PathBuilder.Make(), finished: Skia.PathBuilder.Make(),
      empty: Skia.PathBuilder.Make().build(),
    }));
    const livePath = useSharedValue(geometry.empty);
    const finishedPath = useSharedValue(geometry.empty);
    const dirty = useSharedValue(false);
    const lastDiagnosticsAt = useSharedValue(0);
    const lastReported = useSharedValue<Diagnostics>({ rate: 0, delta: null });

    const finishStroke = () => {
      'worklet';
      if (recorder.value.writer === null) return;
      recorder.value.writer = null;
      geometry.finished.addPath(geometry.live.build());
      geometry.live.reset();
      livePath.value = geometry.empty;
      finishedPath.value = geometry.finished.build();
      dirty.value = false;
    };

    // 开源库 RNGH 负责分发原始触摸事件，Reanimated/Worklets 将回调放在 UI 线程。
    // 每次只认落笔时选定的触点编号，防止第二根手指让笔迹跳到另一个位置。
    const gesture = useManualGesture({
      shouldCancelWhenOutside: false,
      onTouchesDown: (event) => {
        'worklet';
        const now = performance.now();
        const state = recorder.value;
        const touch = event.changedTouches[0];
        if (state.writer !== null || !touch) return;
        beginStroke(state, touch.id);
        appendPoint(state, touch.absoluteX, touch.absoluteY, now);
        geometry.live.moveTo(touch.x, touch.y);
        // 零长度线段使用圆端帽，小数点等轻点也能留下墨迹。
        geometry.live.lineTo(touch.x, touch.y);
        dirty.value = true;
        GestureStateManager.activate(event.handlerTag);
      },
      onTouchesMove: (event) => {
        'worklet';
        const now = performance.now();
        const state = recorder.value;
        for (const touch of event.changedTouches) {
          if (touch.id === state.writer) {
            appendPoint(state, touch.absoluteX, touch.absoluteY, now);
            geometry.live.lineTo(touch.x, touch.y);
            dirty.value = true;
          }
        }
      },
      onTouchesUp: (event) => {
        'worklet';
        const now = performance.now();
        const state = recorder.value;
        for (const touch of event.changedTouches) {
          if (touch.id === state.writer) {
            appendPoint(state, touch.absoluteX, touch.absoluteY, now);
            geometry.live.lineTo(touch.x, touch.y);
            finishStroke();
          }
        }
        if (state.writer === null) GestureStateManager.deactivate(event.handlerTag);
      },
      onTouchesCancel: (event) => {
        'worklet';
        for (const touch of event.changedTouches) {
          if (touch.id === recorder.value.writer) finishStroke();
        }
      },
      onFinalize: () => {
        'worklet';
        // 系统打断或取消时保留已经送达的点；结束回调不制造额外的采样点。
        finishStroke();
      },
    });

    useFrameCallback(useCallback(() => {
      'worklet';
      if (dirty.value) {
        livePath.value = geometry.live.build();
        dirty.value = false;
      }
      const now = performance.now();
      if (now - lastDiagnosticsAt.value < 125) return;
      lastDiagnosticsAt.value = now;
      const next = readDiagnostics(recorder.value, now);
      if (next.rate !== lastReported.value.rate || next.delta !== lastReported.value.delta) {
        lastReported.value = next;
        scheduleOnRN(onDiagnostics, next);
      }
    }, [dirty, geometry, lastDiagnosticsAt, lastReported, livePath, onDiagnostics, recorder]));

    return (
      <GestureDetector gesture={gesture}>
        <Canvas style={styles.surface}>
          <Path path={finishedPath} {...pen} />
          <Path path={livePath} {...pen} />
        </Canvas>
      </GestureDetector>
    );
  },
));

const pen = {
  color: '#23352f', style: 'stroke', strokeWidth: 2.25,
  strokeCap: 'round', strokeJoin: 'round',
} as const;

const styles = StyleSheet.create({ surface: { flex: 1 } });
