import { StyleSheet, Text, View, type TextStyle } from 'react-native';
import {
  parseFormulaNodes,
  parseMathSegments,
  type MathNode,
} from '../utils/mathFormat';

type FormulaProps = {
  latex: string;
  fontSize?: number;
  color?: string;
  display?: boolean;
};

type RichTextProps = {
  text: string;
  fontSize?: number;
  color?: string;
  textStyle?: TextStyle;
};

function renderNodes(
  nodes: MathNode[],
  fontSize: number,
  color: string,
  keyPrefix: string,
): React.ReactNode[] {
  const subSize = Math.max(10, Math.round(fontSize * 0.68));
  return nodes.map((node, idx) => {
    const key = `${keyPrefix}_${idx}`;
    if (node.kind === 'text') {
      return (
        <Text key={key} style={[styles.mathText, { fontSize, color }]}>
          {node.value}
        </Text>
      );
    }

    if (node.kind === 'frac') {
      return (
        <View key={key} style={styles.fracCol}>
          <View style={styles.fracRow}>
            {renderNodes(node.num, Math.max(11, fontSize - 2), color, `${key}_n`)}
          </View>
          <View style={[styles.fracBar, { backgroundColor: color }]} />
          <View style={styles.fracRow}>
            {renderNodes(node.den, Math.max(11, fontSize - 2), color, `${key}_d`)}
          </View>
        </View>
      );
    }

    if (node.kind === 'sqrt') {
      return (
        <View key={key} style={styles.sqrtWrap}>
          <Text style={[styles.mathText, { fontSize: fontSize + 1, color }]}>√</Text>
          <View style={[styles.sqrtBody, { borderTopColor: color }]}>
            {renderNodes(node.body, fontSize, color, `${key}_r`)}
          </View>
        </View>
      );
    }

    if (node.kind === 'sup') {
      return (
        <View key={key} style={styles.scriptGroup}>
          {renderNodes(node.base, fontSize, color, `${key}_b`)}
          <View style={[styles.supBox, { marginBottom: Math.round(fontSize * 0.42) }]}>
            {renderNodes(node.sup, subSize, color, `${key}_u`)}
          </View>
        </View>
      );
    }

    if (node.kind === 'sub') {
      return (
        <View key={key} style={styles.scriptGroup}>
          {renderNodes(node.base, fontSize, color, `${key}_b`)}
          <View style={[styles.subBox, { marginTop: Math.round(fontSize * 0.38) }]}>
            {renderNodes(node.sub, subSize, color, `${key}_l`)}
          </View>
        </View>
      );
    }

    if (node.kind === 'subsup') {
      return (
        <View key={key} style={styles.scriptGroup}>
          {renderNodes(node.base, fontSize, color, `${key}_b`)}
          <View style={styles.subsupCol}>
            <View style={styles.inlineRow}>
              {renderNodes(node.sup, subSize, color, `${key}_u`)}
            </View>
            <View style={styles.inlineRow}>
              {renderNodes(node.sub, subSize, color, `${key}_l`)}
            </View>
          </View>
        </View>
      );
    }

    if (node.kind === 'bigop') {
      const isLim = node.op === 'lim';
      if (isLim) {
        return (
          <View key={key} style={styles.limCol}>
            <Text style={[styles.mathText, { fontSize, color, fontWeight: '600' }]}>lim</Text>
            {node.sub && (
              <View style={styles.inlineRow}>
                {renderNodes(node.sub, subSize, color, `${key}_limsub`)}
              </View>
            )}
          </View>
        );
      }
      return (
        <View key={key} style={styles.scriptGroup}>
          <Text style={[styles.bigOp, { fontSize: Math.round(fontSize * 1.45), color }]}>
            {node.op}
          </Text>
          {(node.sup || node.sub) && (
            <View style={styles.subsupCol}>
              <View style={styles.inlineRow}>
                {node.sup ? renderNodes(node.sup, subSize, color, `${key}_bu`) : <Text style={{ fontSize: subSize }}> </Text>}
              </View>
              <View style={styles.inlineRow}>
                {node.sub ? renderNodes(node.sub, subSize, color, `${key}_bl`) : <Text style={{ fontSize: subSize }}> </Text>}
              </View>
            </View>
          )}
        </View>
      );
    }

    if (node.kind === 'matrix') {
      const showLeft = node.left !== '';
      const showRight = node.right !== '';
      return (
        <View key={key} style={styles.matrixWrap}>
          {showLeft && (
            <View
              style={[
                styles.delimLeft,
                node.left === '[' && { borderTopWidth: 1.5, borderBottomWidth: 1.5, borderColor: color },
                node.left === '|' && { borderLeftWidth: 1.5, borderColor: color },
                node.left === '(' && { borderLeftWidth: 1.5, borderRadius: 8, borderColor: color },
                node.left === '{' && { borderLeftWidth: 1.5, borderColor: color },
                node.left === '[' && { borderLeftWidth: 1.5 },
              ]}
            />
          )}
          <View style={styles.matrixGrid}>
            {node.rows.map((row, rIdx) => (
              <View key={`${key}_r${rIdx}`} style={styles.matrixRow}>
                {row.map((cell, cIdx) => (
                  <View key={`${key}_c${rIdx}_${cIdx}`} style={styles.matrixCell}>
                    {renderNodes(cell, Math.max(12, fontSize - 1), color, `${key}_${rIdx}_${cIdx}`)}
                  </View>
                ))}
              </View>
            ))}
          </View>
          {showRight && (
            <View
              style={[
                styles.delimRight,
                node.right === ']' && { borderTopWidth: 1.5, borderBottomWidth: 1.5, borderColor: color },
                node.right === '|' && { borderRightWidth: 1.5, borderColor: color },
                node.right === ')' && { borderRightWidth: 1.5, borderRadius: 8, borderColor: color },
                node.right === '}' && { borderRightWidth: 1.5, borderColor: color },
                node.right === ']' && { borderRightWidth: 1.5 },
              ]}
            />
          )}
        </View>
      );
    }

    return null;
  });
}

/** 渲染单个纯 LaTeX 公式块（支持分式横线、积分上下限、极限下方、矩阵对齐） */
export function MathFormulaView({
  latex,
  fontSize = 15,
  color = '#1b3a2e',
  display = false,
}: FormulaProps) {
  const nodes = parseFormulaNodes(latex);
  return (
    <View style={[styles.formulaRow, display && styles.displayRow]}>
      {renderNodes(nodes, fontSize, color, 'f')}
    </View>
  );
}

/** 将含任意 $...$ / $$...$$ / \(...\) 混排的中英文数学文本渲染为易读排版视图 */
export function MathRichText({
  text,
  fontSize = 15.5,
  color = '#22392b',
  textStyle,
}: RichTextProps) {
  const segments = parseMathSegments(text);
  if (segments.length === 0 || !segments.some((s) => s.type !== 'text')) {
    return (
      <Text style={[{ fontSize, color, lineHeight: Math.round(fontSize * 1.45) }, textStyle]}>
        {text}
      </Text>
    );
  }

  return (
    <View style={styles.richContainer}>
      {segments.map((seg, idx) => {
        if (seg.type === 'text') {
          return (
            <Text
              key={`seg_${idx}`}
              style={[{ fontSize, color, lineHeight: Math.round(fontSize * 1.45) }, textStyle]}
            >
              {seg.raw}
            </Text>
          );
        }
        if (seg.type === 'display_math') {
          return (
            <View key={`seg_${idx}`} style={styles.displayPill}>
              <MathFormulaView latex={seg.raw} fontSize={fontSize + 1} color="#173a2c" display />
            </View>
          );
        }
        return (
          <View key={`seg_${idx}`} style={styles.inlinePill}>
            <MathFormulaView latex={seg.raw} fontSize={fontSize} color="#173a2c" />
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  richContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 3,
  },
  inlinePill: {
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 4,
    backgroundColor: '#e5efe9',
  },
  displayPill: {
    width: '100%',
    paddingVertical: 8,
    paddingHorizontal: 10,
    marginVertical: 3,
    borderRadius: 7,
    backgroundColor: '#eef5f1',
    borderWidth: 1,
    borderColor: '#cfe0d6',
    alignItems: 'center',
  },
  formulaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
  },
  displayRow: {
    justifyContent: 'center',
  },
  inlineRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  mathText: {
    fontWeight: '500',
  },
  fracCol: {
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 3,
  },
  fracRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 3,
  },
  fracBar: {
    height: 1.3,
    alignSelf: 'stretch',
    marginVertical: 2,
  },
  sqrtWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 2,
  },
  sqrtBody: {
    flexDirection: 'row',
    alignItems: 'center',
    borderTopWidth: 1.3,
    paddingTop: 1,
    paddingHorizontal: 2,
  },
  scriptGroup: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  supBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 1,
  },
  subBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 1,
  },
  subsupCol: {
    justifyContent: 'space-between',
    marginLeft: 1,
    marginRight: 2,
  },
  limCol: {
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 4,
  },
  bigOp: {
    fontWeight: '400',
    lineHeight: 28,
  },
  matrixWrap: {
    flexDirection: 'row',
    alignItems: 'stretch',
    marginHorizontal: 3,
    paddingVertical: 2,
  },
  delimLeft: {
    width: 6,
    marginRight: 2,
  },
  delimRight: {
    width: 6,
    marginLeft: 2,
  },
  matrixGrid: {
    justifyContent: 'center',
    gap: 3,
  },
  matrixRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  matrixCell: {
    minWidth: 18,
    alignItems: 'center',
  },
});
