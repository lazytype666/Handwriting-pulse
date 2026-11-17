import { useState } from 'react';
import { Keyboard, KeyboardAvoidingView, Modal, Platform, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ActionButton } from './Controls';

type Props = { question: string; onSave: (value: string) => void; onClose: () => void };

export function QuestionEditor({ question, onSave, onClose }: Props) {
  const [draft, setDraft] = useState(question);
  return (
    <Modal transparent onRequestClose={onClose} statusBarTranslucent navigationBarTranslucent>
      <SafeAreaView style={styles.backdrop}>
        <KeyboardAvoidingView style={styles.available} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={styles.align}>
            <View style={styles.card} accessibilityViewIsModal>
              <Text accessibilityRole="header" style={styles.heading}>题干</Text>
              <Text style={styles.help}>可直接粘贴题目。在线提示会将题干文字发送给配置的模型服务；留空时使用本地提示。</Text>
              <TextInput accessibilityLabel="题干文本" multiline value={draft} onChangeText={setDraft}
                placeholder="在这里粘贴题目" placeholderTextColor="#667875"
                textAlignVertical="top" style={styles.input} />
              <View style={styles.actions}>
                <ActionButton label="取消" onPress={onClose} />
                <ActionButton label="完成" primary onPress={() => { Keyboard.dismiss(); onSave(draft.trim()); }} />
              </View>
            </View>
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(18,35,32,0.30)' }, available: { flex: 1 },
  align: { padding: 12, alignItems: 'center' },
  card: { width: '100%', maxWidth: 620, padding: 16, borderRadius: 10, backgroundColor: '#fff', gap: 10 },
  heading: { fontSize: 19, fontWeight: '700', color: '#203631' },
  help: { color: '#4b605b', fontSize: 14, lineHeight: 21 },
  input: { minHeight: 85, maxHeight: 160, borderRadius: 6, borderWidth: 1, borderColor: '#879b95',
    padding: 10, color: '#1b302a', backgroundColor: '#fcfdfc', fontSize: 16 },
  actions: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'flex-end', gap: 8 },
});
