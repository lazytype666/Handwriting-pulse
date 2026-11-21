import { useRef, useState } from 'react';
import {
  Keyboard,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { SAMPLE_PROBLEMS } from '../engine';
import { IS_EXPO_GO, recognizeQuestion, type OcrSource } from '../ocr';
import { ActionButton } from './Controls';

type Props = { question: string; onSave: (value: string) => void; onClose: () => void };
type Message = { kind: 'info' | 'error'; text: string };

export function QuestionEditor({ question, onSave, onClose }: Props) {
  const [draft, setDraft] = useState(question);
  const [busy, setBusy] = useState<OcrSource | null>(null);
  const [message, setMessage] = useState<Message | null>(null);
  const running = useRef(false);

  async function recognize(source: OcrSource) {
    if (running.current) return;
    running.current = true;
    Keyboard.dismiss();
    setMessage(null);
    setBusy(source);
    try {
      const text = await recognizeQuestion(source);
      if (text === null) return;
      setDraft((previous) => {
        const head = previous.trimEnd();
        return head ? head + '\n\n' + text : text;
      });
      setMessage({ kind: 'info', text: '已填入识别结果。请核对公式、上下标和符号后点「完成」。' });
    } catch (error) {
      setMessage({ kind: 'error', text: error instanceof Error ? error.message : '识别失败，请重试。' });
    } finally {
      running.current = false;
      setBusy(null);
    }
  }

  return (
    <Modal transparent onRequestClose={onClose} statusBarTranslucent navigationBarTranslucent>
      <SafeAreaView style={styles.backdrop}>
        <KeyboardAvoidingView style={styles.available} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={styles.align}>
            <View style={styles.card} accessibilityViewIsModal>
              <Text accessibilityRole="header" style={styles.heading}>
                题干与解题图谱
              </Text>
              <Text style={styles.help}>
                可粘贴任意题目（支持 $...$
                数学式），或拍照/选图后裁剪识别印刷体题目；识题在本机完成，照片不会上传。配置大模型后保存即在后台预解步骤图谱；下方预设示例自带离线步骤图与歧路预警，断网可直接体验。
              </Text>
              <View style={styles.ocrRow}>
                <ActionButton
                  label={busy === 'camera' ? '识别中…' : '拍照识题'}
                  disabled={busy !== null}
                  onPress={() => {
                    void recognize('camera');
                  }}
                />
                <ActionButton
                  label={busy === 'library' ? '识别中…' : '相册识题'}
                  disabled={busy !== null}
                  onPress={() => {
                    void recognize('library');
                  }}
                />
              </View>
              {IS_EXPO_GO && (
                <Text style={styles.notice}>
                  Expo Go 不包含设备端 OCR 模块。请按 README 安装 EAS 开发构建或独立 APK 后使用识题。
                </Text>
              )}
              {message && (
                <Text
                  accessibilityLiveRegion="polite"
                  style={[styles.message, message.kind === 'error' && styles.error]}
                >
                  {message.text}
                </Text>
              )}
              <View style={styles.sampleSection}>
                <Text style={styles.sampleTitle}>快速加载示例题目（离线自带图谱）：</Text>
                <View style={styles.sampleWrap}>
                  {SAMPLE_PROBLEMS.map((sample) => {
                    const active = draft.trim() === sample.problemText;
                    return (
                      <Pressable
                        key={sample.id}
                        accessibilityRole="button"
                        onPress={() => setDraft(sample.problemText)}
                        style={({ pressed }) => [
                          styles.sampleChip,
                          active && styles.sampleChipActive,
                          pressed && styles.pressed,
                        ]}
                      >
                        <Text style={[styles.sampleChipText, active && styles.sampleChipTextActive]}>
                          {sample.category} · {sample.label}
                        </Text>
                      </Pressable>
                    );
                  })}
                </View>
              </View>
              <TextInput
                accessibilityLabel="题干文本"
                multiline
                value={draft}
                onChangeText={setDraft}
                placeholder="在这里粘贴数学题目、拍照识别，或点击上方示例快速填入"
                placeholderTextColor="#667875"
                textAlignVertical="top"
                style={styles.input}
              />
              <View style={styles.actions}>
                <ActionButton label="清空" disabled={busy !== null} onPress={() => setDraft('')} />
                <ActionButton label="取消" onPress={onClose} />
                <ActionButton
                  label="完成"
                  primary
                  disabled={busy !== null}
                  onPress={() => {
                    Keyboard.dismiss();
                    onSave(draft.trim());
                  }}
                />
              </View>
            </View>
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(18,35,32,0.30)' },
  available: { flex: 1 },
  align: { padding: 12, alignItems: 'center' },
  card: {
    width: '100%',
    maxWidth: 620,
    padding: 16,
    borderRadius: 10,
    backgroundColor: '#fff',
    gap: 10,
  },
  heading: { fontSize: 19, fontWeight: '700', color: '#203631' },
  help: { color: '#4b605b', fontSize: 13.5, lineHeight: 20 },
  ocrRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  notice: { color: '#6a5220', backgroundColor: '#fbf3df', borderRadius: 6, padding: 9, fontSize: 13, lineHeight: 19 },
  message: { color: '#245044', fontSize: 13, lineHeight: 19 },
  error: { color: '#822e21' },
  sampleSection: { gap: 6 },
  sampleTitle: { fontSize: 12.5, fontWeight: '600', color: '#2b4740' },
  sampleWrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  sampleChip: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 14,
    backgroundColor: '#eff5f2',
    borderWidth: 1,
    borderColor: '#d0dfd8',
  },
  sampleChipActive: { backgroundColor: '#244642', borderColor: '#244642' },
  sampleChipText: { fontSize: 12, color: '#244038', fontWeight: '600' },
  sampleChipTextActive: { color: '#ffffff' },
  pressed: { opacity: 0.65 },
  input: {
    minHeight: 85,
    maxHeight: 160,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#879b95',
    padding: 10,
    color: '#1b302a',
    backgroundColor: '#fcfdfc',
    fontSize: 15.5,
  },
  actions: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'flex-end', gap: 8 },
});
