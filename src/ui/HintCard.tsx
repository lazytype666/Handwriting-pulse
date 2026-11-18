import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { HINT_STYLE_LABELS } from '../engine';
import type { useHints } from '../hints';
import { ActionButton } from './Controls';
import { STALL_LABEL } from './HistoryView';

type Props = Pick<
  ReturnType<typeof useHints>,
  'current' | 'advance' | 'dismiss' | 'dismissWithThinking'
>;

export function HintCard({ current, advance, dismiss, dismissWithThinking }: Props) {
  if (!current) return null;
  const styleTag = current.style ? ` · ${HINT_STYLE_LABELS[current.style]}` : '';
  const sourceLabel =
    current.source === 'offline'
      ? '当前为离线提示'
      : current.source === 'cache'
        ? '已缓存思路'
        : current.source === 'online'
          ? '题目提示'
          : '正在请求';

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.title}>
          L{current.level} · {STALL_LABEL[current.type]}{styleTag}
        </Text>
        <Text style={styles.source}>{sourceLabel}</Text>
      </View>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.copy}>
        <Text accessibilityLiveRegion="polite" style={styles.text}>{current.text}</Text>
      </ScrollView>
      <View style={styles.actions}>
        <ActionButton label="我在想，别打扰" onPress={dismissWithThinking} />
        <ActionButton label="收起" onPress={dismiss} />
        {current.level < 3 && (
          <ActionButton
            label="再提示一点"
            primary
            onPress={advance}
            disabled={current.source === 'loading'}
          />
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    alignSelf: 'center',
    width: '100%',
    maxWidth: 540,
    maxHeight: '94%',
    padding: 12,
    gap: 5,
    backgroundColor: '#f1f6f3',
    borderColor: '#b4c8bb',
    borderWidth: 1,
    borderRadius: 9,
  },
  header: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 4,
  },
  title: { fontWeight: '700', fontSize: 15, color: '#244636' },
  source: { color: '#53705d', fontSize: 12 },
  scroll: { flexGrow: 0, flexShrink: 1 },
  copy: { paddingVertical: 3 },
  text: { fontSize: 16, color: '#22392b', lineHeight: 23 },
  actions: { flexDirection: 'row', justifyContent: 'flex-end', gap: 6, flexWrap: 'wrap' },
});
