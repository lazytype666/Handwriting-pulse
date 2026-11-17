import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useState } from 'react';
import { StyleSheet } from 'react-native';
import { Canvas, PaintStyle, Path, Picture, Skia, StrokeCap, StrokeJoin, type SkPath } from '@shopify/react-native-skia';
import { GestureDetector, GestureStateManager, useManualGesture } from 'react-native-gesture-handler';
import { useFrameCallback, useSharedValue } from 'react-native-reanimated';
import { runOnUIAsync, scheduleOnRN } from 'react-native-worklets';
import { ERASER_WIDTH, REWRITE_WINDOW_MS, type InkTool, type StallEvent, type Thresholds } from '../stalls/config';
import { cooldownLeft, createDetector, detect } from '../stalls/detection';
import { createFeatures, observeTouch, pauseAt, speedRatioAt } from '../stalls/features';
import { appendPoint, beginStroke, copyStrokes, createRecorder, readDiagnostics, type Diagnostics, type Stroke } from './recorder';

export interface InkCanvasHandle {
  snapshot(): Promise<{ strokes: Stroke[]; stallEvents: StallEvent[] }>;
  undo(): Promise<void>;
}

interface InkCanvasProps {
  tool: InkTool;
  thresholds: Thresholds;
  onStall: (event: StallEvent) => void;
  onDiagnostics: (diagnostics: Diagnostics) => void;
}

const PEN_WIDTH = 2.25;
const INK = '#23352f';
const PAPER = '#ffffff';
const REPORT_INTERVAL_MS = 125;

function makePaint(color: string, width: number) {
  const paint = Skia.Paint();
  paint.setColor(Skia.Color(color));
  paint.setStyle(PaintStyle.Stroke);
  paint.setStrokeWidth(width);
  paint.setStrokeCap(StrokeCap.Round);
  paint.setStrokeJoin(StrokeJoin.Round);
  paint.setAntiAlias(true);
  return paint;
}

function makeGeometry() {
  const picture = Skia.PictureRecorder();
  picture.beginRecording();
  return {
    picture, blank: picture.finishRecordingAsPicture(),
    builder: Skia.PathBuilder.Make(), empty: Skia.PathBuilder.Make().build(),
    pen: makePaint(INK, PEN_WIDTH), eraser: makePaint(PAPER, ERASER_WIDTH),
  };
}

export const InkCanvas = memo(forwardRef<InkCanvasHandle, InkCanvasProps>(
  function InkCanvas({ tool, thresholds, onDiagnostics, onStall }, ref) {
    const recorder = useSharedValue(createRecorder());
    const features = useSharedValue(createFeatures());
    const detector = useSharedValue(createDetector());
    const chosen = useSharedValue(tool);
    const limits = useSharedValue(thresholds);
    const activeTool = useSharedValue<InkTool>('pen');
    useEffect(() => { chosen.value = tool; }, [chosen, tool]);
    useEffect(() => { limits.value = thresholds; }, [limits, thresholds]);

    const [geometry] = useState(makeGeometry);
    const paths = useSharedValue<{ path: SkPath; tool: InkTool }[]>([]);
    const finished = useSharedValue(geometry.blank);
    const livePen = useSharedValue(geometry.empty);
    const liveEraser = useSharedValue(geometry.empty);
    const dirty = useSharedValue(false);
    const lastReportAt = useSharedValue(0);

    const refreshPicture = () => {
      'worklet';
      // 在白纸上按时间顺序重放开源 Skia 路径。橡皮仅覆盖其经过的条带；
      // 撤销橡皮后，被覆盖的笔画从原始路径恢复，之后画的笔可正常盖回。
      const target = geometry.picture.beginRecording();
      for (const saved of paths.value) {
        target.drawPath(saved.path, saved.tool === 'eraser' ? geometry.eraser : geometry.pen);
      }
      finished.value = geometry.picture.finishRecordingAsPicture();
    };

    const finishStroke = () => {
      'worklet';
      if (recorder.value.writer === null) return;
      recorder.value.writer = null;
      paths.value.push({ path: geometry.builder.build(), tool: activeTool.value });
      geometry.builder.reset();
      livePen.value = geometry.empty;
      liveEraser.value = geometry.empty;
      dirty.value = false;
      refreshPicture();
    };

    useImperativeHandle(ref, () => ({
      snapshot: () => runOnUIAsync(() => {
        'worklet';
        // 事件和点都由 UI 线程写入，同一屏障复制二者，不会漏掉 JS 尚未显示的事件。
        return { strokes: copyStrokes(recorder.value), stallEvents: detector.value.events.map(e => ({ ...e })) };
      }),
      undo: () => runOnUIAsync(() => {
        'worklet';
        finishStroke();
        const removed = recorder.value.strokes.pop();
        if (!removed) return;
        paths.value.pop();
        if (removed.tool === 'eraser') {
          features.value.eraseAt = -1;
          features.value.erasePath = [];
        }
        refreshPicture();
      }),
    }));

    const sample = (down: boolean, x: number, y: number, absoluteX: number, absoluteY: number, now: number) => {
      'worklet';
      // RNGH 负责分发，Reanimated/Worklets 在 UI 线程运行。按窗口坐标采集，
      // 用画布局部坐标求弧长和擦除距离，题干展开改变画布位置也不会拼出伪线段。
      const state = recorder.value;
      const f = features.value;
      const currentTool = activeTool.value;
      const moved = !down && (f.lastX !== x || f.lastY !== y);
      appendPoint(state, absoluteX, absoluteY, now);
      const rewrite = observeTouch(f, currentTool, down, x, y, now, limits.value);
      const event = detect(detector.value, f, rewrite, currentTool === 'pen' && moved, now,
        state.lastTimestamp, absoluteX, absoluteY, limits.value);
      if (event) scheduleOnRN(onStall, event);
      if (down) geometry.builder.moveTo(x, y);
      geometry.builder.lineTo(x, y);
      dirty.value = true;
    };

    const gesture = useManualGesture({
      shouldCancelWhenOutside: false,
      onTouchesDown: event => {
        'worklet';
        const now = performance.now();
        const touch = event.changedTouches[0];
        if (recorder.value.writer !== null || !touch) return;
        activeTool.value = chosen.value;
        beginStroke(recorder.value, touch.id, activeTool.value);
        sample(true, touch.x, touch.y, touch.absoluteX, touch.absoluteY, now);
        GestureStateManager.activate(event.handlerTag);
      },
      onTouchesMove: event => {
        'worklet';
        const now = performance.now();
        for (const touch of event.changedTouches) {
          if (touch.id === recorder.value.writer) sample(false, touch.x, touch.y, touch.absoluteX, touch.absoluteY, now);
        }
      },
      onTouchesUp: event => {
        'worklet';
        const now = performance.now();
        for (const touch of event.changedTouches) {
          if (touch.id === recorder.value.writer) {
            sample(false, touch.x, touch.y, touch.absoluteX, touch.absoluteY, now);
            finishStroke();
          }
        }
        if (recorder.value.writer === null) GestureStateManager.deactivate(event.handlerTag);
      },
      onTouchesCancel: event => {
        'worklet';
        for (const touch of event.changedTouches) {
          if (touch.id === recorder.value.writer) finishStroke();
        }
      },
      onFinalize: () => {
        'worklet';
        finishStroke();
      },
    });

    useFrameCallback(useCallback(() => {
      'worklet';
      if (dirty.value) {
        const live = activeTool.value === 'pen' ? livePen : liveEraser;
        live.value = geometry.builder.build();
        dirty.value = false;
      }
      const now = performance.now();
      if (now - lastReportAt.value < REPORT_INTERVAL_MS || features.value.lastTouch < 0) return;
      lastReportAt.value = now;
      const f = features.value;
      const d = detector.value;
      const nowRatio = speedRatioAt(f, now);
      scheduleOnRN(onDiagnostics, {
        ...readDiagnostics(recorder.value, now),
        pauseMs: Math.round(pauseAt(f, now)),
        speedRatio: nowRatio === null ? null : Math.round(nowRatio * 100) / 100,
        rewritten: f.rewrittenAt >= 0 && now - f.rewrittenAt <= REWRITE_WINDOW_MS,
        cooldownMs: Math.ceil(cooldownLeft(d, now, limits.value)),
        lastStall: d.events.length ? d.events[d.events.length - 1].type : null,
      });
    }, [activeTool, detector, dirty, features, geometry, lastReportAt, limits, liveEraser, livePen, onDiagnostics, recorder]));

    return (
      <GestureDetector gesture={gesture}>
        <Canvas style={styles.surface}>
          <Picture picture={finished} />
          <Path path={livePen} style="stroke" color={INK} strokeWidth={PEN_WIDTH} strokeCap="round" strokeJoin="round" />
          <Path path={liveEraser} style="stroke" color={PAPER} strokeWidth={ERASER_WIDTH} strokeCap="round" strokeJoin="round" />
        </Canvas>
      </GestureDetector>
    );
  },
));

const styles = StyleSheet.create({ surface: { flex: 1, backgroundColor: PAPER } });
