import { useMemo } from 'react';
import { FlatList, Modal, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  HINT_STYLES,
  HINT_STYLE_LABELS,
  type PolicySnapshot,
} from '../engine';
import type { StallEvent, StallType } from '../stalls';
import { ActionButton } from './Controls';

export const STALL_LABEL: Record<StallType, string> = {
  pause_erase: '停顿涂改',
  speed_drop: '速度骤降',
  cognitive_stall: '状态机阻滞',
};

interface HistoryViewProps {
  events: StallEvent[];
  policySnapshot?: PolicySnapshot;
  onClose: () => void;
}

export function HistoryView({ events, policySnapshot, onClose }: HistoryViewProps) {
  const latestFirst = useMemo(() => [...events].reverse(), [events]);
  const pauseCount = useMemo(
    () => events.filter(e => e.type === 'pause_erase').length,
    [events],
  );
  const speedCount = useMemo(
    () => events.filter(e => e.type === 'speed_drop').length,
    [events],
  );
  const stateCount = useMemo(
    () => events.filter(e => e.type === 'cognitive_stall').length,
    [events],
  );

  return (
    <Modal onRequestClose={onClose} statusBarTranslucent navigationBarTranslucent>
      <SafeAreaView style={styles.screen}>
        <View style={styles.content}>
          <View style={styles.header}>
            <Text accessibilityRole="header" style={styles.title}>本次书写历史</Text>
            <ActionButton label="返回画布" onPress={onClose} />
          </View>
          <FlatList
            data={latestFirst}
            keyExtractor={event => String(event.id)}
            ListHeaderComponent={
              <View style={styles.headerBlock}>
                <View style={styles.counts}>
                  <Text style={styles.metric}>停顿涂改 {pauseCount} 次</Text>
                  <Text style={styles.metric}>速度骤降 {speedCount} 次</Text>
                  {stateCount > 0 && <Text style={styles.metric}>状态机阻滞 {stateCount} 次</Text>}
                </View>
                {policySnapshot && (
                  <View style={styles.policyBox}>
                    <Text style={styles.policyTitle}>
                      个性化路由 (LinUCB) · 阈值偏置 {policySnapshot.thresholdOffset >= 0 ? '+' : ''}
                      {policySnapshot.thresholdOffset.toFixed(2)}
                    </Text>
                    <View style={styles.styleRow}>
                      {HINT_STYLES.map(s => {
                        const arm = policySnapshot.arms[s];
                        const avg = arm.pulls > 0 ? (arm.rewardSum / arm.pulls).toFixed(2) : '—';
                        return (
                          <Text key={s} style={styles.styleItem}>
                            {HINT_STYLE_LABELS[s]}：{arm.pulls}次（均分 {avg}）
                          </Text>
                        );
                      })}
                    </View>
                  </View>
                )}
              </View>
            }
            ListEmptyComponent={<Text style={styles.empty}>本次书写还没有触发卡壳提示。</Text>}
            renderItem={({ item }) => (
              <View style={styles.row}>
                <Text style={styles.name}>{STALL_LABEL[item.type]}</Text>
                <Text style={styles.time}>
                  {new Date(item.startedAt).toLocaleTimeString('zh-CN', { hour12: false })}
                </Text>
              </View>
            )}
          />
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
  headerBlock: { marginVertical: 14, gap: 10 },
  counts: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
    backgroundColor: '#edf3f0',
    padding: 14,
    borderRadius: 8,
  },
  metric: { fontSize: 15, fontWeight: '600', color: '#244642' },
  policyBox: {
    backgroundColor: '#f6f9f7',
    borderWidth: 1,
    borderColor: '#dce6e0',
    padding: 12,
    borderRadius: 8,
    gap: 6,
  },
  policyTitle: { fontSize: 13, fontWeight: '600', color: '#28443b' },
  styleRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  styleItem: { fontSize: 12, color: '#465c53', fontVariant: ['tabular-nums'] },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
    minHeight: 52,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: '#d4dfda',
    gap: 8,
  },
  name: { color: '#263c35', fontSize: 16 },
  time: { color: '#53655f', fontVariant: ['tabular-nums'] },
  empty: { marginVertical: 22, fontSize: 15, color: '#53655f' },
});
