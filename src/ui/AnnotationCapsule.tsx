import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

type AnnotationAction = 'DEEP_THINKING_HOLD' | 'STUCK_CONCEPT_GAP' | 'STUCK_CALCULATION';

type Props = {
  onAnnotate: (action: AnnotationAction) => void;
  totalAnnotations: number;
  compact?: boolean;
};

/**
 * 零负担伴随式微标注胶囊栏：悬浮于画布右上角，尽量不遮挡中央书写区。
 * 用于在真机演算中一键记录“深度心算”负样本及“定理/算式卡壳”精确触发点。
 */
export function AnnotationCapsule({ onAnnotate, totalAnnotations, compact = false }: Props) {
  const [stuckMenuOpen, setStuckMenuOpen] = useState(false);
  const [flashText, setFlashText] = useState<string | null>(null);

  const triggerAction = (action: AnnotationAction, toast: string) => {
    setStuckMenuOpen(false);
    setFlashText(toast);
    onAnnotate(action);
    setTimeout(() => {
      setFlashText(prev => (prev === toast ? null : prev));
    }, 1400);
  };

  return (
    <View pointerEvents="box-none" style={styles.anchor}>
      <View style={[styles.row, compact && styles.rowCompact]}>
        {flashText && (
          <View style={styles.toast}>
            <Text style={styles.toastText}>{flashText}</Text>
          </View>
        )}
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="标注正在深度思考请勿打扰"
          onPress={() => triggerAction('DEEP_THINKING_HOLD', '已记：心算勿扰')}
          style={({ pressed }) => [styles.pill, pressed && styles.pressed]}
        >
          <Text style={styles.pillText}>心算勿扰</Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="主动求助并标注卡壳类型"
          onPress={() => setStuckMenuOpen(open => !open)}
          style={({ pressed }) => [
            styles.pill,
            styles.stuckPill,
            stuckMenuOpen && styles.stuckActive,
            pressed && styles.pressed,
          ]}
        >
          <Text style={[styles.pillText, stuckMenuOpen && styles.activeText]}>
            我卡住了{totalAnnotations > 0 ? ` (${totalAnnotations})` : ''}
          </Text>
        </Pressable>
      </View>

      {stuckMenuOpen && (
        <View style={[styles.dropdown, compact && styles.dropdownCompact]}>
          <Pressable
            accessibilityRole="button"
            onPress={() => triggerAction('STUCK_CONCEPT_GAP', '已记：思路卡壳')}
            style={({ pressed }) => [styles.menuBtn, pressed && styles.pressed]}
          >
            <Text style={styles.menuText}>定理 / 思路卡壳</Text>
          </Pressable>
          {!compact && <View style={styles.sep} />}
          <Pressable
            accessibilityRole="button"
            onPress={() => triggerAction('STUCK_CALCULATION', '已记：算式受阻')}
            style={({ pressed }) => [styles.menuBtn, pressed && styles.pressed]}
          >
            <Text style={styles.menuText}>算式 / 化简受阻</Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  anchor: { position: 'absolute', top: 8, right: 8, alignItems: 'flex-end', gap: 5 },
  row: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 5 },
  rowCompact: { flexDirection: 'column', alignItems: 'flex-end' },
  pill: {
    minHeight: 36,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 18,
    backgroundColor: 'rgba(239,245,242,0.9)',
    borderWidth: 1,
    borderColor: '#cde0d6',
    justifyContent: 'center',
  },
  stuckPill: { backgroundColor: 'rgba(254,247,237,0.94)', borderColor: '#ebd0a8' },
  stuckActive: { backgroundColor: '#8f4700', borderColor: '#8f4700' },
  pillText: { fontSize: 12, fontWeight: '600', color: '#26443a' },
  activeText: { color: '#ffffff' },
  pressed: { opacity: 0.65 },
  dropdown: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#dce6e0',
    overflow: 'hidden',
    elevation: 3,
  },
  dropdownCompact: { flexDirection: 'column', alignItems: 'stretch' },
  menuBtn: { minHeight: 38, paddingHorizontal: 11, justifyContent: 'center' },
  menuText: { fontSize: 12.5, fontWeight: '600', color: '#244038' },
  sep: { width: 1, height: 20, backgroundColor: '#dce6e0' },
  toast: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    backgroundColor: 'rgba(36,70,66,0.9)',
  },
  toastText: { fontSize: 11.5, color: '#fff', fontWeight: '600' },
});
