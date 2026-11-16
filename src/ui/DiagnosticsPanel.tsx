import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import type { Diagnostics } from '../capture';

type Props = { data: Diagnostics };

export function DiagnosticsPanel({ data }: Props) {
  const [expanded, setExpanded] = useState(true);
  return (
    <View style={styles.panel}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={expanded ? '隐藏采样诊断' : '显示采样诊断'}
        accessibilityState={{ expanded }}
        onPress={() => setExpanded(value => !value)}
        style={styles.toggle}
      >
        <Text style={styles.title}>{expanded ? '采样诊断' : '显示诊断'}</Text>
        {expanded && <Text style={styles.action}>隐藏</Text>}
      </Pressable>
      {expanded && (
        <View style={styles.readings} pointerEvents="none">
          <View style={styles.row}>
            <Text style={styles.label}>近 1 秒</Text>
            <Text style={styles.value}>{data.rate} 点/秒</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>点间隔</Text>
            <Text style={styles.value}>{data.delta === null ? '—' : data.delta + ' ms'}</Text>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  panel: { backgroundColor: 'rgba(241,244,245,0.89)', borderRadius: 8, alignSelf: 'flex-start' },
  toggle: { minHeight: 44, paddingHorizontal: 12, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 16 },
  title: { fontSize: 14, color: '#243638', fontWeight: '600' },
  action: { fontSize: 12, color: '#4d6366' },
  readings: { paddingHorizontal: 12, paddingBottom: 12, gap: 4 },
  row: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between', gap: 12 },
  label: { fontSize: 13, color: '#435456' },
  value: { fontSize: 14, color: '#17292c', fontVariant: ['tabular-nums'] },
});
