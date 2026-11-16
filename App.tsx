import { useRef, useState } from 'react';
import { Platform, Pressable, StatusBar, StyleSheet, Text, View } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { InkCanvas, type Diagnostics, type InkCanvasHandle } from './src/capture';
import { copyToFolder, saveSnapshot } from './src/export';
import { DiagnosticsPanel } from './src/ui/DiagnosticsPanel';
import { ExportDialog, type ExportResult } from './src/ui/ExportDialog';

function describeError(error: unknown) {
  return error instanceof Error ? error.message : '请检查可用存储空间后重试。';
}

export default function App() {
  const canvas = useRef<InkCanvasHandle>(null);
  const fileOperation = useRef(false);
  const [saving, setSaving] = useState(false);
  const [copying, setCopying] = useState(false);
  const [result, setResult] = useState<ExportResult | null>(null);
  const [diagnostics, setDiagnostics] = useState<Diagnostics>({ rate: 0, delta: null });

  async function exportInk() {
    if (fileOperation.current || !canvas.current) return;
    fileOperation.current = true;
    setSaving(true);
    try {
      const strokes = await canvas.current.snapshot();
      const receipt = await saveSnapshot(strokes);
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
    <GestureHandlerRootView style={styles.sheet}>
      <StatusBar hidden />
      <InkCanvas ref={canvas} onDiagnostics={setDiagnostics} />
      <View pointerEvents="box-none" style={styles.controls}>
        <DiagnosticsPanel data={diagnostics} />
      </View>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="导出笔迹"
        accessibilityState={{ busy: saving, disabled: saving }}
        disabled={saving}
        onPress={exportInk}
        style={({ pressed }) => [styles.exportButton, pressed && styles.pressed]}
      >
        <Text style={styles.exportText}>{saving ? '正在导出…' : '导出'}</Text>
      </Pressable>
      {result && (
        <ExportDialog result={result} copying={copying} onCopy={saveCopy}
          onClose={() => { if (!fileOperation.current) setResult(null); }}
        />
      )}
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  sheet: { flex: 1, backgroundColor: '#fff' },
  controls: {
    position: 'absolute', top: Platform.OS === 'ios' ? 48 : Math.max(12, StatusBar.currentHeight ?? 12),
    left: 12, alignItems: 'flex-start',
  },
  exportButton: { position: 'absolute', right: 16, bottom: 44, minHeight: 44, minWidth: 64, paddingHorizontal: 14, alignItems: 'center', justifyContent: 'center', borderRadius: 8, backgroundColor: '#263f41' },
  exportText: { color: '#fff', fontWeight: '600', fontSize: 15 },
  pressed: { opacity: 0.8 },
});
