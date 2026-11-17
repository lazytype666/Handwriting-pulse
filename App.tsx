import { useCallback, useRef, useState } from 'react';
import { StatusBar, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { InkCanvas, type Diagnostics, type InkCanvasHandle, type InkTool } from './src/capture';
import { copyToFolder, saveSnapshot } from './src/export';
import { useHints } from './src/hints';
import { DEFAULT_THRESHOLDS, type StallEvent } from './src/stalls';
import { ActionButton } from './src/ui/Controls';
import { DiagnosticsPanel } from './src/ui/DiagnosticsPanel';
import { ExportDialog, type ExportResult } from './src/ui/ExportDialog';
import { HintCard } from './src/ui/HintCard';
import { HistoryView } from './src/ui/HistoryView';
import { QuestionEditor } from './src/ui/QuestionEditor';

function describeError(error: unknown) {
  return error instanceof Error ? error.message : '请检查可用存储空间后重试。';
}

export default function App() {
  return (
    <GestureHandlerRootView style={styles.sheet}>
      <SafeAreaProvider><Workspace /></SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

function Workspace() {
  const { width, height } = useWindowDimensions();
  const canvas = useRef<InkCanvasHandle>(null);
  const fileOperation = useRef(false);
  const [saving, setSaving] = useState(false);
  const [copying, setCopying] = useState(false);
  const [result, setResult] = useState<ExportResult | null>(null);
  const [diagnostics, setDiagnostics] = useState<Diagnostics>({
    rate: 0, delta: null, pauseMs: 0, rewritten: false, speedRatio: null, cooldownMs: 0, lastStall: null,
  });
  const [thresholds, setThresholds] = useState(DEFAULT_THRESHOLDS);
  const [tool, setTool] = useState<InkTool>('pen');
  const [question, setQuestion] = useState('');
  const [view, setView] = useState<'question' | 'history' | null>(null);
  const [events, setEvents] = useState<StallEvent[]>([]);
  const { current, advance, dismiss, trigger } = useHints(question);
  const onStall = useCallback((event: StallEvent) => {
    setEvents(previous => [...previous, event]);
    trigger(event);
  }, [trigger]);

  async function exportInk() {
    if (fileOperation.current || !canvas.current) return;
    fileOperation.current = true;
    setSaving(true);
    try {
      const { strokes, stallEvents } = await canvas.current.snapshot();
      const receipt = await saveSnapshot(strokes, stallEvents);
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
      setResult({ ...saved, copyError: '另存失败。' + describeError(error) + ' 应用文档目录中已保留原件。' });
    } finally {
      fileOperation.current = false;
      setCopying(false);
    }
  }

  return (
    <SafeAreaView style={styles.sheet}>
      <StatusBar hidden />
      <View style={styles.topbar}>
        <Text accessibilityRole="header" numberOfLines={1} style={styles.title}>笔迹脉搏</Text>
        <View style={styles.topActions}>
          <ActionButton label={question ? '查看题干' : '题干'} onPress={() => setView('question')} />
          <ActionButton label="历史" onPress={() => setView('history')} />
        </View>
        {width >= 600 && !!question && <Text numberOfLines={1} style={styles.preview}>{question}</Text>}
      </View>
      <View style={styles.paper}>
        <InkCanvas ref={canvas} tool={tool} thresholds={thresholds} onStall={onStall} onDiagnostics={setDiagnostics} />
        <DiagnosticsPanel data={diagnostics} thresholds={thresholds} onChange={setThresholds} />
        {current && <View pointerEvents="box-none" style={[styles.hintSpace, { maxHeight: height < 400 ? '94%' : '68%' }]}>
          <HintCard current={current} advance={advance} dismiss={dismiss} />
        </View>}
      </View>
      <View style={styles.toolbar}>
        <View style={styles.tools}>
          <ActionButton label="笔" selected={tool === 'pen'} onPress={() => setTool('pen')} />
          <ActionButton label="橡皮" selected={tool === 'eraser'} onPress={() => setTool('eraser')} />
          <ActionButton label="撤销" onPress={() => { void canvas.current?.undo(); }} />
        </View>
        <ActionButton label={saving ? '正在导出…' : '导出'} accessibilityLabel="导出笔迹" primary disabled={saving} onPress={exportInk} />
      </View>
      {view === 'question' && <QuestionEditor question={question} onClose={() => setView(null)}
        onSave={value => { if (value !== question) dismiss(); setQuestion(value); setView(null); }} />}
      {view === 'history' && <HistoryView events={events} onClose={() => setView(null)} />}
      {result && (
        <ExportDialog result={result} copying={copying} onCopy={saveCopy}
          onClose={() => { if (!fileOperation.current) setResult(null); }}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  sheet: { flex: 1, backgroundColor: '#fff' },
  topbar: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 12, minHeight: 48, gap: 8,
    borderBottomWidth: StyleSheet.hairlineWidth, borderColor: '#ccd8d2' },
  title: { color: '#253f35', fontWeight: '700', fontSize: 18, flexShrink: 1 },
  preview: { flex: 1, color: '#52665d', marginLeft: 8 },
  topActions: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap' },
  paper: { flex: 1, overflow: 'hidden', backgroundColor: '#fff' },
  hintSpace: { position: 'absolute', bottom: 8, left: 8, right: 8, alignItems: 'center' },
  toolbar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap',
    paddingHorizontal: 10, paddingVertical: 4, gap: 4, borderTopWidth: StyleSheet.hairlineWidth, borderColor: '#ccd8d2' },
  tools: { flexDirection: 'row', flexWrap: 'wrap', gap: 3 },
});
