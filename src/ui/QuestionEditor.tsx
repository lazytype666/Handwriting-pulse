import { useMemo, useRef, useState } from 'react';
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
import {
  parseLatexBlocks,
  SAMPLE_PROBLEMS,
  type HandwritingPulseEngine,
} from '../engine';
import { latexToReadableMath } from '../utils/mathFormat';
import {
  captureProblemForIngest,
  hasConfiguredVlmEndpoint,
  IS_EXPO_GO,
  type OcrSource,
} from '../ocr';
import { ActionButton } from './Controls';
import { MathFormulaView, MathRichText } from './MathFormulaText';

type Props = {
  question: string;
  engine?: HandwritingPulseEngine;
  onSave: (value: string) => void;
  onClose: () => void;
};
type Message = { kind: 'info' | 'error'; text: string };

const OCR_SOURCE_LABEL: Record<string, string> = {
  vlm_local: '本地视觉小模型提取 LaTeX',
  vlm_cloud: '云端视觉模型提取 LaTeX',
  mlkit_llm_refined: '端侧 ML Kit + 本地数学小模型公式修复',
  mlkit_local: '设备端 ML Kit 离线识别',
};

export function QuestionEditor({ question, engine, onSave, onClose }: Props) {
  const [draft, setDraft] = useState(question);
  const [busy, setBusy] = useState<OcrSource | null>(null);
  const [message, setMessage] = useState<Message | null>(null);
  const running = useRef(false);

  const previewBlocks = useMemo(() => parseLatexBlocks(draft.trim()), [draft]);
  const cacheOverview = useMemo(
    () => (draft.trim() && engine ? engine.getProblemCacheOverview(draft) : null),
    [draft, engine],
  );
  const vlmReady = hasConfiguredVlmEndpoint();

  async function recognize(source: OcrSource) {
    if (running.current) return;
    running.current = true;
    Keyboard.dismiss();
    setMessage(null);
    setBusy(source);
    try {
      const captured = await captureProblemForIngest(source);
      if (captured === null) return;

      let recognizedText = captured.mlkitText ?? '';
      let sourceNote = OCR_SOURCE_LABEL.mlkit_local;

      if (engine) {
        if (captured.mode === 'vlm_only' || captured.mode === 'mlkit_with_vlm') {
          const ingested = await engine.ingestProblemDraftFromImage(
            captured.uri,
            captured.mlkitText,
          );
          recognizedText = ingested.problemText;
          sourceNote =
            (ingested.ocrSource && OCR_SOURCE_LABEL[ingested.ocrSource]) || sourceNote;
        } else if (captured.mlkitText) {
          const ingested = await engine.ingestProblemDraftFromOcr(captured.mlkitText);
          recognizedText = ingested.problemText;
          sourceNote =
            (ingested.ocrSource && OCR_SOURCE_LABEL[ingested.ocrSource]) || sourceNote;
        }
      }

      if (!recognizedText.trim()) {
        throw new Error('未能提取出有效题干，请让题目占满画面或保持清晰后重试。');
      }

      setDraft((previous) => {
        const head = previous.trimEnd();
        return head ? head + '\n\n' + recognizedText : recognizedText;
      });
      setMessage({
        kind: 'info',
        text: `已通过「${sourceNote}」填入预览。请核对公式与上下标后点「完成」，系统将在后台静默生成解题图谱并封存答案。`,
      });
    } catch (error) {
      setMessage({
        kind: 'error',
        text: error instanceof Error ? error.message : '识别失败，请重试。',
      });
    } finally {
      running.current = false;
      setBusy(null);
    }
  }

  return (
    <Modal transparent onRequestClose={onClose} statusBarTranslucent navigationBarTranslucent>
      <SafeAreaView style={styles.backdrop}>
        <KeyboardAvoidingView
          style={styles.available}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        >
          <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={styles.align}>
            <View style={styles.card} accessibilityViewIsModal>
              <Text accessibilityRole="header" style={styles.heading}>
                题干与解题图谱
              </Text>
              <Text style={styles.help}>
                可粘贴任意题目（支持 $...$ 数学式），或拍照/选图裁剪识别；确认保存后在 5~30 秒审题期间由本地数学强化小模型/云端自动预解步骤图谱并封存答案。
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
              {IS_EXPO_GO && !vlmReady && (
                <Text style={styles.notice}>
                  Expo Go 不包含设备端 ML Kit OCR。请使用 EAS 开发构建/APK，或在 .env.local 配置本地/云端 VLM（EXPO_PUBLIC_VLM_MODEL）直接识题。
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
                        <Text
                          style={[
                            styles.sampleChipText,
                            active && styles.sampleChipTextActive,
                          ]}
                        >
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
              {previewBlocks.length > 0 && (
                <View style={styles.previewBox}>
                  <Text style={styles.previewTitle}>
                    题干排版预览（核对公式无误后点「完成」）
                  </Text>
                  <View style={styles.blockWrap}>
                    {previewBlocks.map((b, idx) =>
                      b.type === 'text' ? (
                        <Text key={idx} style={styles.previewText}>
                          {b.content}
                        </Text>
                      ) : (
                        <View
                          key={idx}
                          style={[
                            styles.formulaChip,
                            b.type === 'display_formula' && styles.displayFormulaChip,
                          ]}
                        >
                          <MathFormulaView
                            latex={b.content}
                            fontSize={b.type === 'display_formula' ? 16 : 14.5}
                            display={b.type === 'display_formula'}
                          />
                        </View>
                      ),
                    )}
                  </View>
                </View>
              )}
              {cacheOverview?.cached && (
                <View style={styles.cacheBox}>
                  <Text style={styles.cacheTitle}>
                    解题图谱已缓存（
                    {cacheOverview.sourceMode === 'edge_offline'
                      ? '本地数学模型/离线图谱'
                      : '云端预温'}
                    ）：{cacheOverview.stepCount} 步路线 · {cacheOverview.pitfallCount} 条歧路
                    {cacheOverview.hasSealedAnswer ? ' · 答案已封存' : ''}
                  </Text>
                  {cacheOverview.stepGoals.slice(0, 3).map((goal, idx) => (
                    <MathRichText
                      key={idx}
                      text={`· 步骤 ${idx + 1}：${latexToReadableMath(goal)}`}
                      fontSize={12.5}
                      color="#31584c"
                    />
                  ))}
                </View>
              )}
              <View style={styles.actions}>
                <ActionButton
                  label="清空"
                  disabled={busy !== null}
                  onPress={() => setDraft('')}
                />
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
  notice: {
    color: '#6a5220',
    backgroundColor: '#fbf3df',
    borderRadius: 6,
    padding: 9,
    fontSize: 13,
    lineHeight: 19,
  },
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
  previewBox: {
    padding: 9,
    borderRadius: 6,
    backgroundColor: '#f6faf8',
    borderWidth: 1,
    borderColor: '#d6e4dd',
    gap: 5,
  },
  previewTitle: { fontSize: 12, fontWeight: '700', color: '#2b5245' },
  blockWrap: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: 4 },
  previewText: { fontSize: 13, color: '#23352f', lineHeight: 19 },
  formulaChip: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    backgroundColor: '#e8f1ec',
    borderWidth: 1,
    borderColor: '#bdd2c8',
  },
  displayFormulaChip: {
    width: '100%',
    alignItems: 'center',
    paddingVertical: 8,
  },
  formulaText: { fontSize: 12.5, color: '#1b4034', fontWeight: '600' },
  cacheBox: {
    padding: 9,
    borderRadius: 6,
    backgroundColor: '#eef6f2',
    borderWidth: 1,
    borderColor: '#b7d2c5',
    gap: 2,
  },
  cacheTitle: { fontSize: 12.5, fontWeight: '700', color: '#1e4d40' },
  cacheStep: { fontSize: 12, color: '#31584c' },
  actions: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'flex-end', gap: 8 },
});
