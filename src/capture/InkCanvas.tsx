import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useState } from 'react';
import { StyleSheet } from 'react-native';
import {
  Canvas,
  PaintStyle,
  Path,
  Picture,
  Skia,
  StrokeCap,
  StrokeJoin,
  type SkPath,
} from '@shopify/react-native-skia';
import { GestureDetector, GestureStateManager, useManualGesture } from 'react-native-gesture-handler';
import { useFrameCallback, useSharedValue } from 'react-native-reanimated';
import { runOnUIAsync, scheduleOnRN } from 'react-native-worklets';
import { ERASER_WIDTH, REWRITE_WINDOW_MS, type InkTool, type StallEvent, type Thresholds } from '../stalls/config';
import { cooldownLeft, createDetector, detect } from '../stalls/detection';
import { createFeatures, observeTouch, pauseAt, speedRatioAt } from '../stalls/features';
import {
  appendPoint,
  beginStroke,
  copyStrokes,
  createRecorder,
  readDiagnostics,
  type Diagnostics,
  type Stroke,
} from './recorder';

export interface CanvasHistory {
  canUndo: boolean;
  canRedo: boolean;
}

export interface InkCanvasHandle {
  snapshot(): Promise<{ strokes: Stroke[]; stallEvents: StallEvent[] }>;
  undo(): Promise<void>;
  redo(): Promise<void>;
  clear(): Promise<void>;
}

interface InkCanvasProps {
  tool: InkTool;
  thresholds: Thresholds;
  onStall: (event: StallEvent) => void;
  onDiagnostics: (diagnostics: Diagnostics) => void;
  onStrokeComplete?: (stroke: Stroke) => void;
  onStrokeUndo?: (stroke: Stroke) => void;
  onHistory: (history: CanvasHistory) => void;
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
    picture,
    blank: picture.finishRecordingAsPicture(),
    builder: Skia.PathBuilder.Make(),
    empty: Skia.PathBuilder.Make().build(),
    pen: makePaint(INK, PEN_WIDTH),
    eraser: makePaint(PAPER, ERASER_WIDTH),
  };
}

export const InkCanvas = memo(
  forwardRef<InkCanvasHandle, InkCanvasProps>(function InkCanvas(
    { tool, thresholds, onDiagnostics, onStall, onStrokeComplete, onStrokeUndo, onHistory },
    ref,
  ) {
    const recorder = useSharedValue(createRecorder());
    const features = useSharedValue(createFeatures());
    const detector = useSharedValue(createDetector());
    const chosen = useSharedValue(tool);
    const limits = useSharedValue(thresholds);
    const activeTool = useSharedValue<InkTool>('pen');
    useEffect(() => {
      chosen.value = tool;
    }, [chosen, tool]);
    useEffect(() => {
      limits.value = thresholds;
    }, [limits, thresholds]);

    const [geometry] = useState(makeGeometry);
    const paths = useSharedValue<{ path: SkPath; tool: InkTool }[]>([]);
    // 撤销时同时保留下一步可恢复的路径和采集记录。新笔画落笔后清空。
    const redoStack = useSharedValue<{ path: SkPath; tool: InkTool; stroke: Stroke }[]>([]);
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
      paths.value.push({
        path: geometry.builder.build(),
        tool: activeTool.value,
      });
      geometry.builder.reset();
      livePen.value = geometry.empty;
      liveEraser.value = geometry.empty;
      dirty.value = false;
      refreshPicture();
      if (onStrokeComplete && recorder.value.strokes.length > 0) {
        const latest = recorder.value.strokes[recorder.value.strokes.length - 1];
        if (latest.points.length > 0) {
          scheduleOnRN(onStrokeComplete, {
            id: latest.id,
            ...(latest.tool ? { tool: latest.tool } : {}),
            points: latest.points.map((point) => ({
              x: point.x,
              y: point.y,
              t: point.t,
            })),
          });
        }
      }
    };

    // 工具按钮只在 JS 线程渲染，撤销/重做可用性由画布在每次结构变化后回报。
    const reportHistory = () => {
      'worklet';
      scheduleOnRN(onHistory, {
        canUndo: recorder.value.strokes.length > 0,
        canRedo: redoStack.value.length > 0,
      });
    };

    useImperativeHandle(
      ref,
      () => ({
        snapshot: () =>
          runOnUIAsync(() => {
            'worklet';
            // 事件和点都由 UI 线程写入，同一屏障复制二者，不会漏掉 JS 尚未显示的事件。
            return {
              strokes: copyStrokes(recorder.value),
              stallEvents: detector.value.events.map((e) => ({ ...e })),
            };
          }),
        undo: () =>
          runOnUIAsync(() => {
            'worklet';
            finishStroke();
            const removed = recorder.value.strokes.pop();
            if (!removed) return;
            const saved = paths.value.pop();
            if (saved) redoStack.value.push({ ...saved, stroke: removed });
            if (removed.tool === 'eraser') {
              features.value.eraseAt = -1;
              features.value.erasePath = [];
            }
            refreshPicture();
            reportHistory();
            if (onStrokeUndo) {
              scheduleOnRN(onStrokeUndo, {
                id: removed.id,
                ...(removed.tool ? { tool: removed.tool } : {}),
                points: removed.points.map((point) => ({
                  x: point.x,
                  y: point.y,
                  t: point.t,
                })),
              });
            }
          }),
        redo: () =>
          runOnUIAsync(() => {
            'worklet';
            // 没有可重做的笔画时直接返回，避免误结束仍在书写的当前笔。
            const saved = redoStack.value.pop();
            if (!saved) return;
            finishStroke();
            recorder.value.strokes.push(saved.stroke);
            paths.value.push({ path: saved.path, tool: saved.tool });
            // 画布检测特征不因重做回滚；但引擎需要重新看到这枚笔画，保持会话与导出一致。
            refreshPicture();
            reportHistory();
            if (onStrokeComplete) {
              scheduleOnRN(onStrokeComplete, {
                id: saved.stroke.id,
                ...(saved.stroke.tool ? { tool: saved.stroke.tool } : {}),
                points: saved.stroke.points.map((point) => ({
                  x: point.x,
                  y: point.y,
                  t: point.t,
                })),
              });
            }
          }),
        clear: () =>
          runOnUIAsync(() => {
            'worklet';
            // 清空画布但不删除已观测的卡壳事件，历史回看与导出仍保留这些记录。
            recorder.value = createRecorder();
            features.value = createFeatures();
            paths.value = [];
            redoStack.value = [];
            const d = detector.value;
            d.lastAt = -1;
            d.slowSince = -1;
            d.slowTriggered = false;
            geometry.builder.reset();
            livePen.value = geometry.empty;
            liveEraser.value = geometry.empty;
            dirty.value = false;
            refreshPicture();
            scheduleOnRN(onDiagnostics, {
              rate: 0,
              delta: null,
              pauseMs: 0,
              rewritten: false,
              speedRatio: null,
              cooldownMs: 0,
              lastStall: d.events.length ? d.events[d.events.length - 1].type : null,
            });
            reportHistory();
          }),
      }),
      [onHistory, onStrokeComplete, onStrokeUndo],
    );

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
      const event = detect(
        detector.value,
        f,
        rewrite,
        currentTool === 'pen' && moved,
        now,
        state.lastTimestamp,
        absoluteX,
        absoluteY,
        limits.value,
      );
      if (event) scheduleOnRN(onStall, event);
      if (down) geometry.builder.moveTo(x, y);
      geometry.builder.lineTo(x, y);
      dirty.value = true;
    };

    const gesture = useManualGesture({
      shouldCancelWhenOutside: false,
      onTouchesDown: (event) => {
        'worklet';
        const now = performance.now();
        const touch = event.changedTouches[0];
        if (recorder.value.writer !== null || !touch) return;
        activeTool.value = chosen.value;
        beginStroke(recorder.value, touch.id, activeTool.value);
        // 新的落笔形成新分支，之前撤销的笔画不再可重做。
        redoStack.value = [];
        reportHistory();
        sample(true, touch.x, touch.y, touch.absoluteX, touch.absoluteY, now);
        GestureStateManager.activate(event.handlerTag);
      },
      onTouchesMove: (event) => {
        'worklet';
        const now = performance.now();
        for (const touch of event.changedTouches) {
          if (touch.id === recorder.value.writer)
            sample(false, touch.x, touch.y, touch.absoluteX, touch.absoluteY, now);
        }
      },
      onTouchesUp: (event) => {
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
      onTouchesCancel: (event) => {
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

    useFrameCallback(
      useCallback(() => {
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
      }, [
        activeTool,
        detector,
        dirty,
        features,
        geometry,
        lastReportAt,
        limits,
        liveEraser,
        livePen,
        onDiagnostics,
        recorder,
      ]),
    );

    return (
      <GestureDetector gesture={gesture}>
        <Canvas style={styles.surface}>
          <Picture picture={finished} />
          <Path
            path={livePen}
            style="stroke"
            color={INK}
            strokeWidth={PEN_WIDTH}
            strokeCap="round"
            strokeJoin="round"
          />
          <Path
            path={liveEraser}
            style="stroke"
            color={PAPER}
            strokeWidth={ERASER_WIDTH}
            strokeCap="round"
            strokeJoin="round"
          />
        </Canvas>
      </GestureDetector>
    );
  }),
);

const styles = StyleSheet.create({
  surface: { flex: 1, backgroundColor: PAPER },
});
