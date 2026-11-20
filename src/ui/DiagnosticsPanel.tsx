import { useState } from 'react';
import { Modal, Pressable, ScrollView, StyleSheet, Text, View, type GestureResponderEvent } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { Diagnostics } from '../capture';
import {
  COGNITIVE_CATEGORY_ZH,
  COMPANION_MODE_LABELS,
  type CognitiveState,
  type CompanionMode,
} from '../engine';
import type { Thresholds } from '../stalls';
import { ActionButton } from './Controls';
import { STALL_LABEL } from './HistoryView';

type Props = {
  data: Diagnostics;
  thresholds: Thresholds;
  onChange: (next: Thresholds) => void;
  cognitiveState?: CognitiveState;
  interventionThreshold?: number;
  companionMode?: CompanionMode;
  learnedCount?: number;
  onChangeCompanionMode?: (mode: CompanionMode) => void;
  onResetPersonalization?: () => void;
};

type SliderProps = {
  label: string; value: number; min: number; max: number; step: number;
  format: (value: number) => string; onChange: (value: number) => void;
};

const MODES: CompanionMode[] = ['zen', 'socratic', 'coach'];

function ThresholdSlider({ label, value, min, max, step, format, onChange }: SliderProps) {
  const [width, setWidth] = useState(0);
  const update = (raw: number) => {
    const quantized = min + Math.round((raw - min) / step) * step;
    onChange(Number(Math.max(min, Math.min(max, quantized)).toFixed(3)));
  };
  const position = (e: GestureResponderEvent) => {
    if (width > 0) update(min + (max - min) * (e.nativeEvent.locationX - 10) / Math.max(1, width - 20));
  };
  return (
    <View>
      <Text style={styles.sliderLabel}>{label}　{format(value)}</Text>
      <View accessible accessibilityRole="adjustable" accessibilityLabel={label}
        accessibilityValue={{ min, max, now: value, text: format(value) }}
        accessibilityActions={[{ name: 'increment' }, { name: 'decrement' }]}
        onAccessibilityAction={e => update(value + (e.nativeEvent.actionName === 'increment' ? step : -step))}
        onLayout={e => setWidth(e.nativeEvent.layout.width)}
        onStartShouldSetResponder={() => true} onResponderTerminationRequest={() => false}
        onResponderGrant={position} onResponderMove={position} style={styles.trackTouch}>
        <View pointerEvents="none" style={styles.track}>
          <View style={[styles.thumb, { left: `${100 * (value - min) / (max - min)}%` }]} />
        </View>
      </View>
    </View>
  );
}

const seconds = (ms: number) => (ms / 1000).toFixed(1) + ' 秒';
const percent = (ratio: number) => Math.round(ratio * 100) + '%';

export function DiagnosticsPanel({
  data,
  thresholds,
  onChange,
  cognitiveState,
  interventionThreshold,
  companionMode = 'socratic',
  learnedCount = 0,
  onChangeCompanionMode,
  onResetPersonalization,
}: Props) {
  const [visible, setVisible] = useState(true);
  const [details, setDetails] = useState(false);
  const [features, setFeatures] = useState(false);

  return (
    <View style={styles.corner}>
      {visible ? (
        <View style={styles.block}>
          <View style={styles.badge}>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="采样诊断与调参"
              onPress={() => setDetails(true)}
              style={styles.stats}
            >
              <Text style={styles.label}>采样诊断　›</Text>
              <Text style={styles.small}>近 1 秒　{data.rate} 点/秒</Text>
              <Text style={styles.small}>点间隔　{data.delta === null ? '—' : data.delta + ' ms'}</Text>
            </Pressable>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="隐藏采样诊断"
              onPress={() => setVisible(false)}
              style={styles.hide}
            >
              <Text style={styles.small}>隐藏</Text>
            </Pressable>
          </View>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={features ? '收起实时特征' : '展开实时特征'}
            accessibilityState={{ expanded: features }}
            style={styles.featureToggle}
            onPress={() => setFeatures(!features)}
          >
            <Text style={styles.small}>{features ? '收起特征　⌃' : '实时特征　⌄'}</Text>
          </Pressable>
          {features && (
            <View pointerEvents="none" style={styles.featureRows}>
              <Text style={styles.small}>停顿　{seconds(data.pauseMs)}</Text>
              <Text style={styles.small}>擦后重写　{data.rewritten ? '是' : '否'}</Text>
              <Text style={styles.small}>速度比　{data.speedRatio === null ? '—' : percent(data.speedRatio)}</Text>
              <Text style={styles.small}>触发　{data.lastStall ? STALL_LABEL[data.lastStall] : '无'}</Text>
              <Text style={styles.small}>冷却　{seconds(data.cooldownMs)}</Text>
              {cognitiveState && (
                <>
                  <Text style={styles.small}>
                    状态　{COGNITIVE_CATEGORY_ZH[cognitiveState.cognitiveCategory]}（{cognitiveState.neuralMode === 'neural_ts' ? '神经网' : '启发式'}）
                  </Text>
                  <Text style={styles.small}>
                    动量/阻力/对齐　{cognitiveState.momentumM.toFixed(2)} / {cognitiveState.resistanceR.toFixed(2)} / {cognitiveState.alignmentA.toFixed(2)}
                  </Text>
                  {!!cognitiveState.recognizedLatex && (
                    <Text numberOfLines={1} style={styles.small}>
                      语法　{cognitiveState.recognizedLatex}
                    </Text>
                  )}
                </>
              )}
            </View>
          )}
        </View>
      ) : (
        <View style={styles.badge}>
          <ActionButton label="显示诊断" onPress={() => setVisible(true)} />
        </View>
      )}
      {details && (
        <Modal transparent onRequestClose={() => setDetails(false)} statusBarTranslucent navigationBarTranslucent>
          <SafeAreaView style={styles.backdrop}>
            <View style={styles.card}>
              <View style={styles.header}>
                <Text accessibilityRole="header" style={styles.title}>采样、状态机与调参</Text>
                <ActionButton label="返回画布" onPress={() => setDetails(false)} />
              </View>
              <ScrollView contentContainerStyle={styles.body}>
                <Text style={styles.readout}>采样　{data.rate} 点/秒　　点间隔　{data.delta ?? '—'} ms</Text>
                <Text style={styles.readout}>当前停顿　{seconds(data.pauseMs)}</Text>
                <Text style={styles.readout}>同区域擦后重写　{data.rewritten ? '是' : '否'}</Text>
                <Text style={styles.readout}>当前 / 基线速度　{data.speedRatio === null ? '积累数据中' : percent(data.speedRatio)}</Text>
                <Text style={styles.readout}>最近触发　{data.lastStall ? STALL_LABEL[data.lastStall] : '尚未触发'}</Text>
                <Text style={styles.readout}>冷却剩余　{seconds(data.cooldownMs)}</Text>
                {cognitiveState && (
                  <>
                    <Text style={styles.readout}>
                      端侧分类　{COGNITIVE_CATEGORY_ZH[cognitiveState.cognitiveCategory]}（{cognitiveState.neuralMode === 'neural_ts' ? '双流量化权重' : '热身启发式'}）
                    </Text>
                    <Text style={styles.readout}>
                      认知状态 [M, R, A]　{cognitiveState.momentumM.toFixed(2)} / {cognitiveState.resistanceR.toFixed(2)} / {cognitiveState.alignmentA.toFixed(2)}
                    </Text>
                    <Text style={styles.readout}>
                      干预倾向 P(t)　{cognitiveState.interventionScore.toFixed(2)} / 阈值 {(interventionThreshold ?? 0.62).toFixed(2)}（惊奇 S={cognitiveState.surpriseScore.toFixed(2)}）
                    </Text>
                    <Text style={styles.readout}>
                      语法对齐　{cognitiveState.matchedStepIndex >= 0 ? `第 ${cognitiveState.matchedStepIndex + 1} 步` : '自由推导'} · {cognitiveState.recognizedLatex || '等待落笔'}
                    </Text>
                  </>
                )}

                <View style={styles.divider} />
                <ThresholdSlider label="停顿阈值" value={thresholds.stallMs} min={500} max={8000} step={100} format={seconds}
                  onChange={stallMs => onChange({ ...thresholds, stallMs })} />
                <ThresholdSlider label="骤降阈值" value={thresholds.speedRatio} min={0.1} max={0.8} step={0.05} format={percent}
                  onChange={speedRatio => onChange({ ...thresholds, speedRatio })} />
                <ThresholdSlider label="冷却时间" value={thresholds.cooldownMs} min={0} max={60000} step={1000} format={seconds}
                  onChange={cooldownMs => onChange({ ...thresholds, cooldownMs })} />

                {onChangeCompanionMode && (
                  <>
                    <View style={styles.divider} />
                    <View style={styles.modeHeader}>
                      <Text style={styles.sliderLabel}>个性化伴随模式（已学 {learnedCount} 次）</Text>
                      {onResetPersonalization && (
                        <Pressable accessibilityRole="button" onPress={onResetPersonalization}>
                          <Text style={styles.resetLink}>重置习惯</Text>
                        </Pressable>
                      )}
                    </View>
                    <View style={styles.modeRow}>
                      {MODES.map(mode => (
                        <ActionButton
                          key={mode}
                          label={COMPANION_MODE_LABELS[mode]}
                          selected={companionMode === mode}
                          onPress={() => onChangeCompanionMode(mode)}
                        />
                      ))}
                    </View>
                  </>
                )}
              </ScrollView>
            </View>
          </SafeAreaView>
        </Modal>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  corner: { position: 'absolute', top: 8, left: 8 },
  badge: { flexDirection: 'row', alignItems: 'flex-start', backgroundColor: 'rgba(239,245,242,0.88)', borderRadius: 7 },
  block: { backgroundColor: 'rgba(239,245,242,0.85)', borderRadius: 7 },
  featureToggle: { minHeight: 30, justifyContent: 'center', paddingHorizontal: 10 },
  featureRows: { paddingHorizontal: 10, paddingBottom: 8, gap: 2 },
  stats: { paddingLeft: 10, paddingVertical: 7, gap: 2 },
  label: { fontWeight: '700', color: '#28443b', fontSize: 13 },
  small: { color: '#465c53', fontVariant: ['tabular-nums'], fontSize: 12 },
  hide: { minWidth: 44, minHeight: 44, alignItems: 'center', justifyContent: 'center' },
  backdrop: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 12, backgroundColor: 'rgba(18,35,32,0.30)' },
  card: { backgroundColor: '#fff', maxHeight: '96%', width: '100%', maxWidth: 460, borderRadius: 10, flexShrink: 1 },
  header: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 14 },
  title: { color: '#243b32', fontSize: 18, fontWeight: '700' },
  body: { paddingHorizontal: 16, paddingBottom: 12, gap: 4 },
  readout: { color: '#354d42', lineHeight: 22, fontVariant: ['tabular-nums'] },
  divider: { height: StyleSheet.hairlineWidth, backgroundColor: '#c6d9ce', marginVertical: 6 },
  sliderLabel: { fontSize: 14, fontWeight: '600', color: '#264234' },
  trackTouch: { height: 40, paddingHorizontal: 10, justifyContent: 'center' },
  track: { height: 4, borderRadius: 2, backgroundColor: '#cad9d2' },
  thumb: { position: 'absolute', top: -7, marginLeft: -9, width: 18, height: 18, borderRadius: 9, backgroundColor: '#32604f' },
  modeHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 2 },
  resetLink: { fontSize: 12.5, color: '#8f4700', fontWeight: '600' },
  modeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 4 },
});
