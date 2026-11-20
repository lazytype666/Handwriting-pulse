import { useState } from 'react';
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
import { ActionButton } from './Controls';

type Props = { question: string; onSave: (value: string) => void; onClose: () => void };

export function QuestionEditor({ question, onSave, onClose }: Props) {
  const [draft, setDraft] = useState(question);

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
                可粘贴任意题目（支持 $...$ 数学式）。配置大模型后保存即在后台预解步骤图谱；下方预设示例自带离线步骤图与歧路预警，断网可直接体验。
              </Text>
              <View style={styles.sampleSection}>
                <Text style={styles.sampleTitle}>快速加载示例题目（离线自带图谱）：</Text>
                <View style={styles.sampleWrap}>
                  {SAMPLE_PROBLEMS.map(sample => {
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
                placeholder="在这里粘贴数学题目，或点击上方示例快速填入"
                placeholderTextColor="#667875"
                textAlignVertical="top"
                style={styles.input}
              />
              <View style={styles.actions}>
                <ActionButton label="清空" onPress={() => setDraft('')} />
                <ActionButton label="取消" onPress={onClose} />
                <ActionButton
                  label="完成"
                  primary
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
