/**
 * 全大学数学 CTC 符号词表与二维笔画关系词表（自动由 ml/export/export_ts_weights.py 与 Python 对齐生成）。
 */
export const MATH_VOCAB: readonly string[] = ["<blank>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "+", "-", "=", "(", ")", "[", "]", "{", "}", "|", "/", ",", ".", "_", "^", "~", "<", ">", "!", "\\frac", "\\sqrt", "\\int", "\\oint", "\\sum", "\\prod", "\\partial", "\\nabla", "\\infty", "\\lim", "\\to", "\\Rightarrow", "\\equiv", "\\neq", "\\approx", "\\le", "\\ge", "\\times", "\\cdot", "\\pm", "\\alpha", "\\beta", "\\gamma", "\\delta", "\\varepsilon", "\\theta", "\\lambda", "\\mu", "\\rho", "\\sigma", "\\phi", "\\Phi", "\\pi", "\\Omega", "\\Delta", "\\Lambda", "\\cap", "\\cup", "\\varnothing", "\\bar", "\\hat", "\\det", "\\matrix"];
export const MATH_VOCAB_SIZE = 125;

export const MATH_VOCAB_INDEX: ReadonlyMap<string, number> = new Map(
  MATH_VOCAB.map((tok, idx) => [tok, idx]),
);

export const EDGE_RELATIONS: readonly string[] = ["NONE", "SAME_SYMBOL", "RIGHT", "ABOVE", "BELOW", "SUP", "SUB"];
export type StrokeEdgeRelation = 'NONE' | 'SAME_SYMBOL' | 'RIGHT' | 'ABOVE' | 'BELOW' | 'SUP' | 'SUB';

export interface CtcDecodeOptions {
  /** 题谱先验浅融合偏置向量（长度 125，按当前题干与步骤期望符号赋予对数概率加成） */
  priorLogitBias?: Float32Array;
  /** 是否启用 TAMER 风格的括号/上下标语法栈约束与树闭合修复 */
  enforceGrammar?: boolean;
}

const ID_LPAREN = MATH_VOCAB.indexOf('(');
const ID_RPAREN = MATH_VOCAB.indexOf(')');
const ID_LBRACK = MATH_VOCAB.indexOf('[');
const ID_RBRACK = MATH_VOCAB.indexOf(']');
const ID_LBRACE = MATH_VOCAB.indexOf('{');
const ID_RBRACE = MATH_VOCAB.indexOf('}');
const ID_SUB = MATH_VOCAB.indexOf('_');
const ID_SUP = MATH_VOCAB.indexOf('^');

/**
 * 从 CTC 帧级 Logits (frames × vocabSize) 提取 125 维符号边缘后验概率 P_ctc(v) ∈ [0, 1]^125。
 * 供下游解题图谱步骤 Logits 与易错点检测 Logits 执行连续软概率内积。
 */
export function computeCtcTokenPosteriors(
  ctcLogits: Float32Array,
  frames: number,
  vocabSize: number,
): Float32Array {
  const posteriors = new Float32Array(vocabSize);
  if (frames <= 0 || vocabSize <= 0) return posteriors;
  const logNotSeen = new Float32Array(vocabSize);
  const maxFrameProb = new Float32Array(vocabSize);
  for (let t = 0; t < frames; t++) {
    const off = t * vocabSize;
    let maxV = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      const val = ctcLogits[off + v];
      if (val > maxV) maxV = val;
    }
    let sumExp = 0;
    for (let v = 0; v < vocabSize; v++) sumExp += Math.exp(ctcLogits[off + v] - maxV);
    const invZ = 1 / Math.max(1e-12, sumExp);
    for (let v = 1; v < vocabSize; v++) {
      const p = Math.min(0.9999, Math.max(0, Math.exp(ctcLogits[off + v] - maxV) * invZ));
      if (p > maxFrameProb[v]) maxFrameProb[v] = p;
      logNotSeen[v] += Math.log(Math.max(1e-6, 1 - p * 0.65));
    }
  }
  for (let v = 1; v < vocabSize; v++) {
    const noisyOr = 1 - Math.exp(logNotSeen[v]);
    posteriors[v] = Number(Math.min(1, 0.55 * maxFrameProb[v] + 0.45 * noisyOr).toFixed(4));
  }
  return posteriors;
}

function grammarTransitionPenalty(prevIds: number[], nextId: number): number {
  const prevId = prevIds.length > 0 ? prevIds[prevIds.length - 1] : -1;
  if ((nextId === ID_SUP || nextId === ID_SUB) && (prevId === ID_SUP || prevId === ID_SUB)) {
    return -3.2;
  }
  if (
    prevIds.length === 0 &&
    (nextId === ID_SUP || nextId === ID_SUB || nextId === ID_RPAREN || nextId === ID_RBRACK || nextId === ID_RBRACE)
  ) {
    return -3.5;
  }
  if (nextId === ID_RPAREN || nextId === ID_RBRACK || nextId === ID_RBRACE) {
    const openTarget = nextId === ID_RPAREN ? ID_LPAREN : nextId === ID_RBRACK ? ID_LBRACK : ID_LBRACE;
    let depth = 0;
    for (const id of prevIds) {
      if (id === openTarget) depth += 1;
      else if (id === nextId) depth -= 1;
    }
    if (depth <= 0) return -3.4;
    if (prevId === openTarget) return -2.4;
  }
  return 0;
}

function computeUnclosedDelimiterCount(ids: number[]) {
  let braceDepth = 0;
  let parenDepth = 0;
  let bracketDepth = 0;
  for (const id of ids) {
    if (id === ID_LBRACE) braceDepth = Math.max(0, braceDepth + 1);
    else if (id === ID_RBRACE) braceDepth = Math.max(0, braceDepth - 1);
    else if (id === ID_LPAREN) parenDepth = Math.max(0, parenDepth + 1);
    else if (id === ID_RPAREN) parenDepth = Math.max(0, parenDepth - 1);
    else if (id === ID_LBRACK) bracketDepth = Math.max(0, bracketDepth + 1);
    else if (id === ID_RBRACK) bracketDepth = Math.max(0, bracketDepth - 1);
  }
  const last = ids.length > 0 ? ids[ids.length - 1] : -1;
  return { braceDepth, parenDepth, bracketDepth, danglingScript: last === ID_SUP || last === ID_SUB };
}

export function repairBalancedTokens(ids: number[]): number[] {
  const out = [...ids];
  while (out.length > 0 && (out[out.length - 1] === ID_SUP || out[out.length - 1] === ID_SUB)) out.pop();
  const st = computeUnclosedDelimiterCount(out);
  for (let i = 0; i < st.braceDepth && i < 3; i++) out.push(ID_RBRACE);
  for (let i = 0; i < st.parenDepth && i < 2; i++) out.push(ID_RPAREN);
  for (let i = 0; i < st.bracketDepth && i < 2; i++) out.push(ID_RBRACK);
  return out;
}

export function formatTokensToLatex(tokens: string[]): string {
  const pieces: string[] = [];
  for (const tok of tokens) {
    const prevTok = pieces.length > 0 ? pieces[pieces.length - 1] : '';
    if (tok.startsWith('\\') || (prevTok.startsWith('\\') && /^[a-z0-9]+$/i.test(tok))) {
      pieces.push(pieces.length > 0 ? ' ' + tok : tok);
    } else {
      pieces.push(tok);
    }
  }
  return pieces.join('').trim();
}

export function decodeCtcGreedy(predIds: number[]): string {
  const tokens: string[] = [];
  let prev = -1;
  for (const idx of predIds) {
    if (idx > 0 && idx !== prev && idx < MATH_VOCAB.length) tokens.push(MATH_VOCAB[idx]);
    prev = idx;
  }
  return formatTokensToLatex(tokens);
}

function logAddExp(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));
}

/**
 * 轻量 CTC Prefix Beam Search 解码（与 Python `decode_ctc_beam_search` 严格等价，含 TAMER 语法栈与题谱先验浅融合）。
 */
export function decodeCtcBeamSearch(
  ctcLogits: Float32Array,
  frames: number,
  vocabSize: number,
  beamWidth = 4,
  topK = 6,
  options?: CtcDecodeOptions,
): string {
  if (frames <= 0 || vocabSize <= 0) return '';
  type BeamEntry = { ids: number[]; pb: number; pnb: number };
  let beams = new Map<string, BeamEntry>([['', { ids: [], pb: 0, pnb: -Infinity }]]);
  const logProbs = new Float32Array(vocabSize);
  const kEff = Math.min(topK, vocabSize);
  const topIds = new Int32Array(kEff);
  const topVals = new Float32Array(kEff);
  const prior = options?.priorLogitBias && options.priorLogitBias.length === vocabSize ? options.priorLogitBias : null;
  const useGrammar = Boolean(options?.enforceGrammar);

  for (let t = 0; t < frames; t++) {
    const off = t * vocabSize;
    let maxV = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      const val = ctcLogits[off + v] + (prior && v > 0 ? prior[v] : 0);
      if (val > maxV) maxV = val;
    }
    let sumExp = 0;
    for (let v = 0; v < vocabSize; v++) {
      sumExp += Math.exp(ctcLogits[off + v] + (prior && v > 0 ? prior[v] : 0) - maxV);
    }
    const logZ = maxV + Math.log(Math.max(1e-12, sumExp));
    for (let v = 0; v < vocabSize; v++) {
      logProbs[v] = ctcLogits[off + v] + (prior && v > 0 ? prior[v] : 0) - logZ;
    }

    topIds.fill(0);
    topVals.fill(-Infinity);
    for (let v = 0; v < vocabSize; v++) {
      const lp = logProbs[v];
      if (lp > topVals[kEff - 1]) {
        let pos = kEff - 1;
        while (pos > 0 && lp > topVals[pos - 1]) {
          topVals[pos] = topVals[pos - 1];
          topIds[pos] = topIds[pos - 1];
          pos--;
        }
        topVals[pos] = lp;
        topIds[pos] = v;
      }
    }
    const candidates: number[] = [];
    let hasBlank = false;
    for (let i = 0; i < kEff; i++) {
      if (topIds[i] === 0) hasBlank = true;
      candidates.push(topIds[i]);
    }
    if (!hasBlank) candidates.push(0);

    const nextBeams = new Map<string, BeamEntry>();
    const upsert = (key: string, ids: number[], addPb: number, addPnb: number) => {
      const prev = nextBeams.get(key);
      if (!prev) nextBeams.set(key, { ids, pb: addPb, pnb: addPnb });
      else {
        if (addPb !== -Infinity) prev.pb = logAddExp(prev.pb, addPb);
        if (addPnb !== -Infinity) prev.pnb = logAddExp(prev.pnb, addPnb);
      }
    };

    for (const [key, entry] of beams.entries()) {
      const pTot = logAddExp(entry.pb, entry.pnb);
      const lastTok = entry.ids.length > 0 ? entry.ids[entry.ids.length - 1] : -1;
      for (const idx of candidates) {
        const lp = logProbs[idx];
        if (idx === 0) {
          upsert(key, entry.ids, pTot + lp, -Infinity);
        } else if (idx === lastTok) {
          upsert(key, entry.ids, -Infinity, entry.pnb + lp);
          const gPen = useGrammar ? grammarTransitionPenalty(entry.ids, idx) : 0;
          upsert(key ? `${key},${idx}` : String(idx), [...entry.ids, idx], -Infinity, entry.pb + lp + gPen);
        } else {
          const gPen = useGrammar ? grammarTransitionPenalty(entry.ids, idx) : 0;
          upsert(key ? `${key},${idx}` : String(idx), [...entry.ids, idx], -Infinity, pTot + lp + gPen);
        }
      }
    }
    const ranked = Array.from(nextBeams.entries())
      .sort((a, b) => logAddExp(b[1].pb, b[1].pnb) - logAddExp(a[1].pb, a[1].pnb))
      .slice(0, beamWidth);
    beams = new Map(ranked);
  }

  let bestIds: number[] = [];
  let bestScore = -Infinity;
  for (const entry of beams.values()) {
    let sc = logAddExp(entry.pb, entry.pnb);
    if (useGrammar) {
      const unclosed = computeUnclosedDelimiterCount(entry.ids);
      sc -= (unclosed.braceDepth + unclosed.parenDepth + unclosed.bracketDepth) * 1.15 + (unclosed.danglingScript ? 1.8 : 0);
    }
    if (sc > bestScore) {
      bestScore = sc;
      bestIds = entry.ids;
    }
  }
  const finalIds = useGrammar ? repairBalancedTokens(bestIds) : bestIds;
  const tokens: string[] = [];
  for (const idx of finalIds) {
    if (idx > 0 && idx < MATH_VOCAB.length) tokens.push(MATH_VOCAB[idx]);
  }
  return formatTokensToLatex(tokens);
}
