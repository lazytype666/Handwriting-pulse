import { Modal, Pressable, StyleSheet, Text, View } from 'react-native';

type Props = {
  title: string;
  message: string;
  confirmLabel: string;
  cancelLabel?: string;
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
};

export function ConfirmDialog({
  title,
  message,
  confirmLabel,
  cancelLabel = '取消',
  destructive,
  onConfirm,
  onCancel,
}: Props) {
  return (
    <Modal transparent animationType="fade" onRequestClose={onCancel} statusBarTranslucent>
      <View style={styles.backdrop}>
        <View style={styles.dialog} accessibilityViewIsModal>
          <Text accessibilityRole="header" style={styles.title}>
            {title}
          </Text>
          <Text style={styles.message}>{message}</Text>
          <View style={styles.actions}>
            <Pressable accessibilityRole="button" onPress={onCancel} style={styles.secondary}>
              <Text style={styles.secondaryText}>{cancelLabel}</Text>
            </Pressable>
            <Pressable
              accessibilityRole="button"
              onPress={onConfirm}
              style={[styles.primary, destructive && styles.destructive]}
            >
              <Text style={styles.primaryText}>{confirmLabel}</Text>
            </Pressable>
          </View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    padding: 20,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(13,24,26,0.32)',
  },
  dialog: { width: '100%', maxWidth: 420, padding: 22, borderRadius: 12, backgroundColor: '#ffffff' },
  title: { fontSize: 19, fontWeight: '700', color: '#18282b' },
  message: { marginTop: 10, color: '#49595a', lineHeight: 22, fontSize: 15 },
  actions: { flexDirection: 'row', justifyContent: 'flex-end', flexWrap: 'wrap', gap: 8, marginTop: 20 },
  secondary: {
    minHeight: 44,
    minWidth: 88,
    paddingHorizontal: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#b6c7c7',
    borderRadius: 6,
  },
  secondaryText: { color: '#203a3b', fontWeight: '600' },
  primary: {
    minHeight: 44,
    minWidth: 88,
    paddingHorizontal: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#263f41',
    borderRadius: 6,
  },
  destructive: { backgroundColor: '#8a3029' },
  primaryText: { color: '#ffffff', fontWeight: '600' },
});
