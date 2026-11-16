import { Modal, Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import type { ExportReceipt } from '../export';

export type ExportResult =
  | { kind: 'saved'; receipt: ExportReceipt; copyUri?: string; copyError?: string }
  | { kind: 'error'; message: string };

type Props = {
  result: ExportResult;
  copying: boolean;
  onCopy: () => void;
  onClose: () => void;
};

export function ExportDialog({ result, copying, onCopy, onClose }: Props) {
  return (
    <Modal transparent animationType="fade" onRequestClose={onClose} statusBarTranslucent>
      <View style={styles.backdrop}>
        <View style={styles.dialog} accessibilityViewIsModal>
          <ScrollView contentContainerStyle={styles.body} bounces={false}>
            <Text accessibilityRole="header" style={styles.title}>
              {result.kind === 'saved' ? '笔迹已导出' : '导出失败'}
            </Text>
            {result.kind === 'saved' ? (
              <>
                <Text style={styles.summary}>{result.receipt.strokeCount} 笔 · {result.receipt.pointCount} 个采样点</Text>
                <Text style={styles.caption}>应用文档目录</Text>
                <Text selectable style={styles.path}>{result.receipt.uri}</Text>
                {result.copyUri && (
                  <>
                    <Text style={styles.caption}>已另存到所选目录</Text>
                    <Text selectable style={styles.path}>{result.copyUri}</Text>
                  </>
                )}
                {result.copyError && <Text selectable style={styles.error}>{result.copyError}</Text>}
                {Platform.OS === 'android' && (
                  <Pressable accessibilityRole="button" disabled={copying} onPress={onCopy} style={styles.secondary}>
                    <Text style={styles.secondaryText}>{copying ? '正在另存…' : '另存到手机文件夹'}</Text>
                  </Pressable>
                )}
              </>
            ) : <Text selectable style={styles.error}>{result.message}</Text>}
            <Pressable accessibilityRole="button" disabled={copying} onPress={onClose} style={styles.primary}>
              <Text style={styles.primaryText}>返回画布</Text>
            </Pressable>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, padding: 20, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(13,24,26,0.32)' },
  dialog: { width: '100%', maxWidth: 440, maxHeight: '88%', borderRadius: 12, backgroundColor: '#ffffff', overflow: 'hidden' },
  body: { padding: 22 },
  title: { fontSize: 21, fontWeight: '700', color: '#18282b' },
  summary: { marginTop: 5, color: '#49595a' },
  caption: { marginTop: 18, marginBottom: 6, color: '#425859', fontSize: 13, fontWeight: '600' },
  path: { color: '#152325', backgroundColor: '#f2f4f4', padding: 10, fontSize: 13 },
  error: { marginVertical: 14, color: '#822e21' },
  secondary: { minHeight: 44, marginTop: 16, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: '#b6c7c7', borderRadius: 6 },
  secondaryText: { color: '#203a3b', fontWeight: '600' },
  primary: { minHeight: 44, marginTop: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: '#263f41', borderRadius: 6 },
  primaryText: { color: '#ffffff', fontWeight: '600' },
});
