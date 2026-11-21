/**
 * 全大学数学 CTC 符号词表（高数 / 线代 / 概率统计，自动由 ml/export/export_ts_weights.py 与 Python 对齐生成）。
 */
export const MATH_VOCAB: readonly string[] = ["<blank>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "+", "-", "=", "(", ")", "[", "]", "{", "}", "|", "/", ",", ".", "_", "^", "~", "<", ">", "!", "\\frac", "\\sqrt", "\\int", "\\oint", "\\sum", "\\prod", "\\partial", "\\nabla", "\\infty", "\\lim", "\\to", "\\Rightarrow", "\\equiv", "\\neq", "\\approx", "\\le", "\\ge", "\\times", "\\cdot", "\\pm", "\\alpha", "\\beta", "\\gamma", "\\delta", "\\varepsilon", "\\theta", "\\lambda", "\\mu", "\\rho", "\\sigma", "\\phi", "\\Phi", "\\pi", "\\Omega", "\\Delta", "\\Lambda", "\\cap", "\\cup", "\\varnothing", "\\bar", "\\hat", "\\det", "\\matrix"];
export const MATH_VOCAB_SIZE = 125;

function formatTokensToLatex(tokens: string[]): string {
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
    if (idx > 0 && idx !== prev && idx < MATH_VOCAB.length) {
      tokens.push(MATH_VOCAB[idx]);
    }
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
 * 轻量 CTC Prefix Beam Search 解码（与 Python `decode_ctc_beam_search` 严格等价）。
 */
export function decodeCtcBeamSearch(
  ctcLogits: Float32Array,
  frames: number,
  vocabSize: number,
  beamWidth = 4,
  topK = 6,
): string {
  if (frames <= 0 || vocabSize <= 0) return '';
  type BeamEntry = { ids: number[]; pb: number; pnb: number };
  let beams = new Map<string, BeamEntry>([['', { ids: [], pb: 0, pnb: -Infinity }]]);
  const logProbs = new Float32Array(vocabSize);
  const kEff = Math.min(topK, vocabSize);
  const topIds = new Int32Array(kEff);
  const topVals = new Float32Array(kEff);

  for (let t = 0; t < frames; t++) {
    const off = t * vocabSize;
    let maxV = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      const val = ctcLogits[off + v];
      if (val > maxV) maxV = val;
    }
    let sumExp = 0;
    for (let v = 0; v < vocabSize; v++) sumExp += Math.exp(ctcLogits[off + v] - maxV);
    const logZ = maxV + Math.log(Math.max(1e-12, sumExp));
    for (let v = 0; v < vocabSize; v++) logProbs[v] = ctcLogits[off + v] - logZ;

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
      if (!prev) {
        nextBeams.set(key, { ids, pb: addPb, pnb: addPnb });
      } else {
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
          const extIds = [...entry.ids, idx];
          const extKey = key ? `${key},${idx}` : String(idx);
          upsert(extKey, extIds, -Infinity, entry.pb + lp);
        } else {
          const extIds = [...entry.ids, idx];
          const extKey = key ? `${key},${idx}` : String(idx);
          upsert(extKey, extIds, -Infinity, pTot + lp);
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
    const sc = logAddExp(entry.pb, entry.pnb);
    if (sc > bestScore) {
      bestScore = sc;
      bestIds = entry.ids;
    }
  }
  const tokens: string[] = [];
  for (const idx of bestIds) {
    if (idx > 0 && idx < MATH_VOCAB.length) tokens.push(MATH_VOCAB[idx]);
  }
  return formatTokensToLatex(tokens);
}
