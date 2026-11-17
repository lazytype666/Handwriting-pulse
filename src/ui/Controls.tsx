import { Pressable, StyleSheet, Text } from 'react-native';

type ButtonProps = {
  label: string;
  onPress: () => void;
  selected?: boolean;
  primary?: boolean;
  disabled?: boolean;
  accessibilityLabel?: string;
};

export function ActionButton({ label, onPress, selected, primary, disabled, accessibilityLabel }: ButtonProps) {
  const filled = primary || selected;
  return (
    <Pressable accessibilityRole="button" accessibilityLabel={accessibilityLabel ?? label}
      accessibilityState={{ disabled: !!disabled, ...(selected === undefined ? {} : { selected }) }}
      onPress={onPress} disabled={disabled}
      style={({ pressed }) => [styles.button, filled && styles.filled, (pressed || disabled) && styles.dim]}>
      <Text style={[styles.label, filled && styles.light]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: { minHeight: 44, minWidth: 44, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 7,
    alignItems: 'center', justifyContent: 'center' },
  filled: { backgroundColor: '#244642' },
  label: { color: '#284440', fontSize: 14, fontWeight: '600' },
  light: { color: '#fff' },
  dim: { opacity: 0.58 },
});
