import { useCallback, useEffect, useRef, useState } from 'react';
import { StatusBar, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import {
  InkCanvas,
  type CanvasHistory,
  type Diagnostics,
  type InkCanvasHandle,
  type InkTool,
  type Stroke,
} from './src/capture';
import {
  HandwritingPulseEngine,
  type AnnotationLabel,
  type CompanionMode,
  type EngineDiagnosticsTick,
} from './src/engine';
import { copyToFolder, saveSnapshot } from './src/export';
import { useHints } from './src/hints';
import { DEFAULT_THRESHOLDS, type StallEvent } from './src/stalls';
import { AnnotationCapsule } from './src/ui/AnnotationCapsule';
import { ConfirmDialog } from './src/ui/ConfirmDialog';
import { ActionButton } from './src/ui/Controls';
import { DiagnosticsPanel } from './src/ui/DiagnosticsPanel';
import { ExportDialog, type ExportResult } from './src/ui/ExportDialog';
import { HintCard } from './src/ui/HintCard';
import { HistoryView } from './src/ui/HistoryView';
import { QuestionEditor } from './src/ui/QuestionEditor';

function describeError(error: unknown) {
  return error instanceof Error ? error.message : '请检查可用存储空间后重试。';
}

function countTotalAnnotations(counts: Record<AnnotationLabel, number>): number {
  return Object.values(counts).reduce((sum, n) => sum + n, 0);
}

export default function App() {
  return (
    <GestureHandlerRootView style={styles.sheet}>
      <SafeAreaProvider>
        <Workspace />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

function Workspace() {
  const { width, height } = useWindowDimensions();
  const canvas = useRef<InkCanvasHandle>(null);
  const fileOperation = useRef(false);
  const engineRef = useRef<HandwritingPulseEngine>(new HandwritingPulseEngine());

  const [saving, setSaving] = useState(false);
  const [copying, setCopying] = useState(false);
  const [result, setResult] = useState<ExportResult | null>(null);
  const [diagnostics, setDiagnostics] = useState<Diagnostics>({
    rate: 0,
    delta: null,
    pauseMs: 0,
    rewritten: false,
    speedRatio: null,
    cooldownMs: 0,
    lastStall: null,
  });
  const [cognitiveTick, setCognitiveTick] = useState<EngineDiagnosticsTick | null>(null);
  const [companionMode, setCompanionMode] = useState<CompanionMode>('socratic');
  const [learnedCount, setLearnedCount] = useState<number>(0);
  const [totalAnnotations, setTotalAnnotations] = useState<number>(0);
  const [thresholds, setThresholds] = useState(DEFAULT_THRESHOLDS);
  const [tool, setTool] = useState<InkTool>('pen');
  const [history, setHistory] = useState<CanvasHistory>({
    canUndo: false,
    canRedo: false,
  });
  const [confirmingClear, setConfirmingClear] = useState(false);
  const [question, setQuestion] = useState('');
  const [prewarmState, setPrewarmState] = useState<'idle' | 'prewarming' | 'ready' | 'offline'>('idle');
  const [view, setView] = useState<'question' | 'history' | null>(null);
  const [events, setEvents] = useState<StallEvent[]>([]);

  // 1. 初始化加载本地持久化运动指纹、个性化权重与预温题干图谱缓存
  useEffect(() => {
    let active = true;
    void engineRef.current.bootstrapStorage().then((profile) => {
      if (!active || !profile) return;
      setCompanionMode(engineRef.current.getCompanionMode());
      setLearnedCount(engineRef.current.getPolicySnapshot().feedbackCount);
    });
    return () => {
      active = false;
    };
  }, []);

  const syncAnnotationCounter = useCallback(() => {
    setTotalAnnotations(countTotalAnnotations(engineRef.current.getAnnotationCounts()));
  }, []);

  const onFeedbackPersist = useCallback(() => {
    const profile = engineRef.current.exportPersistedProfile();
    setLearnedCount(profile.policy.feedbackCount);
    void engineRef.current.persistAllState();
  }, []);

  const { current, advance, dismiss, dismissWithThinking, trigger } = useHints(
    question,
    engineRef.current,
    onFeedbackPersist,
  );

  const onStall = useCallback(
    (event: StallEvent) => {
      setEvents((previous) => [...previous, event]);
      trigger(event, undefined, engineRef.current.getSuggestedHintLevel());
    },
    [trigger],
  );

  // 2. 笔迹落笔完成 / 撤销事件实时进入双流神经语法编码器与因果认知状态网络
  const onStrokeComplete = useCallback(
    (stroke: Stroke) => {
      void engineRef.current
        .onStrokeComplete(stroke)
        .then(() => {
          syncAnnotationCounter();
          setCognitiveTick((prev) =>
            prev
              ? {
                  ...prev,
                  cognitiveState: engineRef.current.getCognitiveState(),
                  suggestedHintLevel: engineRef.current.getSuggestedHintLevel(),
                }
              : {
                  cognitiveState: engineRef.current.getCognitiveState(),
                  interventionThreshold: 0.62,
                  cognitiveStallEvent: null,
                  suggestedHintLevel: engineRef.current.getSuggestedHintLevel(),
                  settledFeedback: null,
                },
          );
        })
        .catch(() => {});
    },
    [syncAnnotationCounter],
  );

  const onStrokeUndo = useCallback(
    (stroke: Stroke) => {
      void engineRef.current
        .onStrokeUndo(stroke)
        .then(() => {
          syncAnnotationCounter();
          setCognitiveTick((prev) =>
            prev
              ? {
                  ...prev,
                  cognitiveState: engineRef.current.getCognitiveState(),
                  suggestedHintLevel: engineRef.current.getSuggestedHintLevel(),
                }
              : null,
          );
        })
        .catch(() => {});
    },
    [syncAnnotationCounter],
  );

  // 3. 诊断采样驱动端侧连续状态机微分更新、自适应traj采样与隐式反馈闭环
  const onDiagnostics = useCallback(
    (data: Diagnostics) => {
      setDiagnostics(data);
      const engine = engineRef.current;
      const tick = engine.processDiagnosticsTick(data, thresholds);
      setCognitiveTick(tick);

      if (tick.cognitiveStallEvent) {
        onStall(tick.cognitiveStallEvent);
      }

      if (tick.settledFeedback) {
        onFeedbackPersist();
        syncAnnotationCounter();
      }
    },
    [thresholds, onStall, onFeedbackPersist, syncAnnotationCounter],
  );

  const handleCompanionAnnotate = useCallback(
    (action: 'DEEP_THINKING_HOLD' | 'STUCK_CONCEPT_GAP' | 'STUCK_CALCULATION') => {
      const outcome = engineRef.current.recordCompanionAnnotation(action);
      syncAnnotationCounter();
      onFeedbackPersist();
      if (outcome.triggerEvent) {
        setEvents((previous) => [...previous, outcome.triggerEvent!]);
        trigger(outcome.triggerEvent, outcome.preferredStyle);
      }
    },
    [syncAnnotationCounter, onFeedbackPersist, trigger],
  );

  async function exportInk() {
    if (fileOperation.current || !canvas.current) return;
    fileOperation.current = true;
    setSaving(true);
    try {
      const { strokes, stallEvents } = await canvas.current.snapshot();
      const summary = engineRef.current.getSummary();
      const receipt = await saveSnapshot(strokes, stallEvents, {
        problemId: summary.problemId,
        problemText: question,
        annotations: summary.annotations,
        stateTrace: summary.stateTrace,
        cognitiveState: summary.latestState,
        latestSyntax: summary.latestSyntax,
        companionMode: summary.companionMode,
        policy: summary.policySnapshot,
        feedbackLogs: summary.feedbackLogs,
      });
      void engineRef.current.persistAllState();
      setResult({ kind: 'saved', receipt });
    } catch (error) {
      setResult({ kind: 'error', message: describeError(error) });
    } finally {
      fileOperation.current = false;
      setSaving(false);
    }
  }

  async function saveCopy() {
    if (result?.kind !== 'saved' || fileOperation.current) return;
    fileOperation.current = true;
    setCopying(true);
    const saved = { ...result, copyError: undefined };
    try {
      const uri = await copyToFolder(saved.receipt);
      setResult(uri ? { ...saved, copyUri: uri } : saved);
    } catch (error) {
      setResult({
        ...saved,
        copyError: '另存失败。' + describeError(error) + ' 应用文档目录中已保留原件。',
      });
    } finally {
      fileOperation.current = false;
      setCopying(false);
    }
  }

  return (
    <SafeAreaView style={styles.sheet}>
      <StatusBar hidden />
      <View style={styles.topbar}>
        <Text accessibilityRole="header" numberOfLines={1} style={styles.title}>
          笔迹脉搏
        </Text>
        <View style={styles.topActions}>
          <ActionButton label={question ? '查看题干' : '题干'} onPress={() => setView('question')} />
          <ActionButton label="历史" onPress={() => setView('history')} />
        </View>
        {!!question && prewarmState !== 'idle' && (
          <View
            style={[
              styles.prewarmBadge,
              prewarmState === 'ready' && styles.prewarmReady,
              prewarmState === 'prewarming' && styles.prewarmBusy,
            ]}
          >
            <Text style={styles.prewarmText}>
              {prewarmState === 'prewarming'
                ? '图谱预解中…'
                : prewarmState === 'ready'
                  ? '图谱已缓存'
                  : '离线兜底'}
            </Text>
          </View>
        )}
        {width >= 600 && !!question && (
          <Text numberOfLines={1} style={styles.preview}>
            {question}
          </Text>
        )}
      </View>
      <View style={styles.paper}>
        <InkCanvas
          ref={canvas}
          tool={tool}
          thresholds={thresholds}
          onStall={onStall}
          onDiagnostics={onDiagnostics}
          onStrokeComplete={onStrokeComplete}
          onStrokeUndo={onStrokeUndo}
          onHistory={setHistory}
        />
        <DiagnosticsPanel
          data={diagnostics}
          thresholds={thresholds}
          onChange={setThresholds}
          cognitiveState={cognitiveTick?.cognitiveState}
          interventionThreshold={cognitiveTick?.interventionThreshold}
          companionMode={companionMode}
          learnedCount={learnedCount}
          onChangeCompanionMode={(mode) => {
            engineRef.current.setCompanionMode(mode);
            setCompanionMode(mode);
            onFeedbackPersist();
          }}
          onResetPersonalization={() => {
            engineRef.current.resetPersonalization();
            setCompanionMode(engineRef.current.getCompanionMode());
            onFeedbackPersist();
          }}
        />
        <AnnotationCapsule
          onAnnotate={handleCompanionAnnotate}
          totalAnnotations={totalAnnotations}
          compact={width < 460}
        />
        {current && (
          <View pointerEvents="box-none" style={[styles.hintSpace, { maxHeight: height < 400 ? '94%' : '68%' }]}>
            <HintCard current={current} advance={advance} dismiss={dismiss} dismissWithThinking={dismissWithThinking} />
          </View>
        )}
      </View>
      <View style={styles.toolbar}>
        <View style={styles.tools}>
          <ActionButton label="笔" selected={tool === 'pen'} onPress={() => setTool('pen')} />
          <ActionButton label="橡皮" selected={tool === 'eraser'} onPress={() => setTool('eraser')} />
          <ActionButton
            label="撤销"
            disabled={!history.canUndo}
            onPress={() => {
              void canvas.current?.undo().catch(() => {});
            }}
          />
          <ActionButton
            label="重做"
            disabled={!history.canRedo}
            onPress={() => {
              void canvas.current?.redo().catch(() => {});
            }}
          />
          <ActionButton
            label="清空"
            accessibilityLabel="清空画布"
            disabled={!history.canUndo}
            onPress={() => setConfirmingClear(true)}
          />
        </View>
        <ActionButton
          label={saving ? '正在导出…' : '导出'}
          accessibilityLabel="导出笔迹"
          primary
          disabled={saving}
          onPress={exportInk}
        />
      </View>
      {view === 'question' && (
        <QuestionEditor
          question={question}
          engine={engineRef.current}
          onClose={() => setView(null)}
          onSave={(value) => {
            if (value !== question) dismiss();
            setQuestion(value);
            setView(null);
            if (!value) {
              setPrewarmState('idle');
              void engineRef.current.updateProblemText(value);
              return;
            }
            setPrewarmState('prewarming');
            void engineRef.current.updateProblemText(value).then((cachedOk) => {
              setPrewarmState(cachedOk ? 'ready' : 'offline');
            });
          }}
        />
      )}
      {view === 'history' && (
        <HistoryView
          events={events}
          policySnapshot={engineRef.current.getPolicySnapshot()}
          annotationCounts={engineRef.current.getAnnotationCounts()}
          onClose={() => setView(null)}
        />
      )}
      {confirmingClear && (
        <ConfirmDialog
          title="清空画布？"
          message="画布上的笔迹会被清除，清空后不可撤销。本次卡壳历史会保留，并可继续导出；认知状态与微标注会随画布一起重置。"
          confirmLabel="清空"
          destructive
          onConfirm={() => {
            setConfirmingClear(false);
            dismiss();
            void canvas.current?.clear();
            engineRef.current.resetSessionTracking();
            setCognitiveTick(null);
            syncAnnotationCounter();
          }}
          onCancel={() => setConfirmingClear(false)}
        />
      )}
      {result && (
        <ExportDialog
          result={result}
          copying={copying}
          onCopy={saveCopy}
          onClose={() => {
            if (!fileOperation.current) setResult(null);
          }}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  sheet: { flex: 1, backgroundColor: '#fff' },
  topbar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    minHeight: 48,
    gap: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: '#ccd8d2',
  },
  title: { color: '#253f35', fontWeight: '700', fontSize: 18, flexShrink: 1 },
  preview: { flex: 1, color: '#52665d', marginLeft: 8 },
  topActions: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap' },
  prewarmBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 12,
    backgroundColor: '#eef3f1',
    borderWidth: 1,
    borderColor: '#ccd8d2',
  },
  prewarmBusy: { backgroundColor: '#fbf5e6', borderColor: '#e5cf9b' },
  prewarmReady: { backgroundColor: '#e9f4ee', borderColor: '#b3d4c2' },
  prewarmText: { fontSize: 11.5, fontWeight: '600', color: '#27473d' },
  paper: { flex: 1, overflow: 'hidden', backgroundColor: '#fff' },
  hintSpace: {
    position: 'absolute',
    bottom: 8,
    left: 8,
    right: 8,
    alignItems: 'center',
  },
  toolbar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    paddingHorizontal: 10,
    paddingVertical: 4,
    gap: 4,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderColor: '#ccd8d2',
  },
  tools: { flexDirection: 'row', flexWrap: 'wrap', gap: 3 },
});
