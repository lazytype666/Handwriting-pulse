import { useMemo } from 'react';
import { FlatList, Modal, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { StallEvent } from '../stalls';
import { ActionButton } from './Controls';

export const STALL_LABEL = { pause_erase: '停顿涂改', speed_drop: '速度骤降' } as const;

export function HistoryView({ events, onClose }: { events: StallEvent[]; onClose: () => void }) {
  const latestFirst = useMemo(() => [...events].reverse(), [events]);
  const pauseCount = useMemo(() => events.filter(e => e.type === 'pause_erase').length, [events]);
  return (
    <Modal onRequestClose={onClose} statusBarTranslucent navigationBarTranslucent>
      <SafeAreaView style={styles.screen}>
        <View style={styles.content}>
          <View style={styles.header}>
            <Text accessibilityRole="header" style={styles.title}>本次书写历史</Text>
            <ActionButton label="返回画布" onPress={onClose} />
          </View>
          <FlatList data={latestFirst} keyExtractor={event => String(event.id)}
            ListHeaderComponent={<View style={styles.counts}>
              <Text style={styles.metric}>停顿涂改 {pauseCount} 次</Text>
              <Text style={styles.metric}>速度骤降 {events.length - pauseCount} 次</Text>
            </View>}
            ListEmptyComponent={<Text style={styles.empty}>本次书写还没有触发卡壳提示。</Text>}
            renderItem={({ item }) => (
              <View style={styles.row}>
                <Text style={styles.name}>{STALL_LABEL[item.type]}</Text>
                <Text style={styles.time}>{new Date(item.startedAt).toLocaleTimeString('zh-CN', { hour12: false })}</Text>
              </View>
            )} />
        </View>
      </SafeAreaView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#fff' },
  content: { flex: 1, width: '100%', maxWidth: 740, alignSelf: 'center', padding: 16 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, flexWrap: 'wrap' },
  title: { fontSize: 21, color: '#203631', fontWeight: '700' },
  counts: { flexDirection: 'row', flexWrap: 'wrap', gap: 16, backgroundColor: '#edf3f0',
    padding: 14, marginVertical: 14, borderRadius: 8 },
  metric: { fontSize: 15, fontWeight: '600', color: '#244642' },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap',
    minHeight: 52, borderBottomWidth: StyleSheet.hairlineWidth, borderColor: '#d4dfda', gap: 8 },
  name: { color: '#263c35', fontSize: 16 }, time: { color: '#53655f', fontVariant: ['tabular-nums'] },
  empty: { marginVertical: 22, fontSize: 15, color: '#53655f' },
});
