/**
 * 纯 TypeScript 数学公式解析与可读化工具（无 React Native 依赖，引擎与 UI 共用）。
 *
 * 将 raw LaTeX 转为：
 * 1. `latexToReadableMath`: 易读的 Unicode 数学文本（用于提示卡片、预览摘要与端侧步骤蒸馏）；
 * 2. `parseMathSegments` / `parseFormulaNodes`: 供 `MathFormulaText` 渲染真分式横线、积分/求和/极限上下限、根号与矩阵排版。
 */

export type MathNode =
  | { kind: 'text'; value: string }
  | { kind: 'frac'; num: MathNode[]; den: MathNode[] }
  | { kind: 'sqrt'; body: MathNode[] }
  | { kind: 'sup'; base: MathNode[]; sup: MathNode[] }
  | { kind: 'sub'; base: MathNode[]; sub: MathNode[] }
  | { kind: 'subsup'; base: MathNode[]; sub: MathNode[]; sup: MathNode[] }
  | { kind: 'bigop'; op: string; sub?: MathNode[]; sup?: MathNode[] }
  | { kind: 'matrix'; left: string; right: string; rows: MathNode[][][] };

export interface RichSegment {
  type: 'text' | 'inline_math' | 'display_math';
  raw: string;
}

const SUP_MAP: Record<string, string> = {
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
  '+': '⁺',
  '-': '⁻',
  '=': '⁼',
  '(': '⁽',
  ')': '⁾',
  a: 'ᵃ',
  b: 'ᵇ',
  c: 'ᶜ',
  d: 'ᵈ',
  e: 'ᵉ',
  f: 'ᶠ',
  g: 'ᵍ',
  h: 'ʰ',
  i: 'ⁱ',
  j: 'ʲ',
  k: 'ᵏ',
  l: 'ˡ',
  m: 'ᵐ',
  n: 'ⁿ',
  o: 'ᵒ',
  p: 'ᵖ',
  r: 'ʳ',
  s: 'ˢ',
  t: 'ᵗ',
  u: 'ᵘ',
  v: 'ᵛ',
  w: 'ʷ',
  x: 'ˣ',
  y: 'ʸ',
  z: 'ᶻ',
};

const SUB_MAP: Record<string, string> = {
  '0': '₀',
  '1': '₁',
  '2': '₂',
  '3': '₃',
  '4': '₄',
  '5': '₅',
  '6': '₆',
  '7': '₇',
  '8': '₈',
  '9': '₉',
  '+': '₊',
  '-': '₋',
  '=': '₌',
  '(': '₍',
  ')': '₎',
  a: 'ₐ',
  e: 'ₑ',
  h: 'ₕ',
  i: 'ᵢ',
  j: 'ⱼ',
  k: 'ₖ',
  l: 'ₗ',
  m: 'ₘ',
  n: 'ₙ',
  o: 'ₒ',
  p: 'ₚ',
  r: 'ᵣ',
  s: 'ₛ',
  t: 'ₜ',
  u: 'ᵤ',
  v: 'ᵥ',
  x: 'ₓ',
};

const LATEX_SYMBOL_MAP: Record<string, string> = {
  alpha: 'α',
  beta: 'β',
  gamma: 'γ',
  delta: 'δ',
  epsilon: 'ε',
  varepsilon: 'ε',
  zeta: 'ζ',
  eta: 'η',
  theta: 'θ',
  vartheta: 'ϑ',
  iota: 'ι',
  kappa: 'κ',
  lambda: 'λ',
  mu: 'μ',
  nu: 'ν',
  xi: 'ξ',
  pi: 'π',
  rho: 'ρ',
  sigma: 'σ',
  tau: 'τ',
  upsilon: 'υ',
  phi: 'φ',
  varphi: 'φ',
  chi: 'χ',
  psi: 'ψ',
  omega: 'ω',
  Gamma: 'Γ',
  Delta: 'Δ',
  Theta: 'Θ',
  Lambda: 'Λ',
  Xi: 'Ξ',
  Pi: 'Π',
  Sigma: 'Σ',
  Phi: 'Φ',
  Psi: 'Ψ',
  Omega: 'Ω',
  infty: '∞',
  to: '→',
  rightarrow: '→',
  leftarrow: '←',
  leftrightarrow: '↔',
  Rightarrow: '⇒',
  implies: '⇒',
  iff: '⇔',
  le: '≤',
  leq: '≤',
  ge: '≥',
  geq: '≥',
  neq: '≠',
  ne: '≠',
  approx: '≈',
  equiv: '≡',
  sim: '∼',
  pm: '±',
  mp: '∓',
  times: '×',
  cdot: '·',
  div: '÷',
  partial: '∂',
  nabla: '∇',
  forall: '∀',
  exists: '∃',
  in: '∈',
  notin: '∉',
  subset: '⊂',
  subseteq: '⊆',
  cup: '∪',
  cap: '∩',
  emptyset: '∅',
  int: '∫',
  iint: '∬',
  iiint: '∭',
  oint: '∮',
  sum: '∑',
  prod: '∏',
  sin: 'sin',
  cos: 'cos',
  tan: 'tan',
  cot: 'cot',
  sec: 'sec',
  csc: 'csc',
  arcsin: 'arcsin',
  arccos: 'arccos',
  arctan: 'arctan',
  sinh: 'sinh',
  cosh: 'cosh',
  ln: 'ln',
  log: 'log',
  exp: 'exp',
  lim: 'lim',
  sup: 'sup',
  inf: 'inf',
  max: 'max',
  min: 'min',
  det: 'det',
  rank: 'rank',
  tr: 'tr',
  dim: 'dim',
  arg: 'arg',
  quad: ' ',
  qquad: '  ',
  dots: '…',
  cdots: '⋯',
  ldots: '…',
  vdots: '⋮',
  ddots: '⋱',
  angle: '∠',
  perp: '⊥',
  parallel: '∥',
  mid: ' | ',
  degree: '°',
};

function toScript(text: string, table: Record<string, string>): string | null {
  const clean = text.trim();
  if (!clean || clean.length > 8) return null;
  let out = '';
  for (const ch of clean) {
    const mapped = table[ch];
    if (!mapped) return null;
    out += mapped;
  }
  return out;
}

function readBracedGroup(src: string, idx: number): { content: string; nextIdx: number } | null {
  let i = idx;
  while (i < src.length && /\s/.test(src[i])) i += 1;
  if (i >= src.length) return null;
  if (src[i] === '{') {
    let depth = 0;
    const start = i + 1;
    for (let j = i; j < src.length; j += 1) {
      if (src[j] === '{') depth += 1;
      else if (src[j] === '}') {
        depth -= 1;
        if (depth === 0) {
          return { content: src.slice(start, j), nextIdx: j + 1 };
        }
      }
    }
    return { content: src.slice(start), nextIdx: src.length };
  }
  if (src[i] === '\\') {
    const m = src.slice(i).match(/^\\([A-Za-z]+|.)/);
    if (m) return { content: m[0], nextIdx: i + m[0].length };
  }
  return { content: src[i], nextIdx: i + 1 };
}

function normalizeLatexCommands(input: string): string {
  return input
    .replace(/\\(?:displaystyle|textstyle|scriptstyle|mathrm|mathbf|mathit|text|boldsymbol)\s*/g, '')
    .replace(/\\(?:left|right|big|Big|bigg|Bigg)\s*([|()\[\]{}.])/g, (_, d: string) =>
      d === '.' ? '' : d,
    )
    .replace(/\\(?:,|;|:|!)/g, ' ')
    .trim();
}

const MATRIX_DELIMS: Record<string, [string, string]> = {
  matrix: ['(', ')'],
  pmatrix: ['(', ')'],
  bmatrix: ['[', ']'],
  Bmatrix: ['{', '}'],
  vmatrix: ['|', '|'],
  Vmatrix: ['‖', '‖'],
  cases: ['{', ''],
  aligned: ['', ''],
};

/** 将纯公式串解析为轻量化结构 AST，供 UI 分式/上下限/根号/矩阵可视化渲染 */
export function parseFormulaNodes(rawLatex: string): MathNode[] {
  const src = normalizeLatexCommands(rawLatex);
  const nodes: MathNode[] = [];
  let i = 0;

  const pushText = (txt: string) => {
    if (!txt) return;
    const last = nodes[nodes.length - 1];
    if (last && last.kind === 'text') {
      last.value += txt;
    } else {
      nodes.push({ kind: 'text', value: txt });
    }
  };

  while (i < src.length) {
    const ch = src[i];

    // 矩阵/分段函数环境：\begin{bmatrix} ... \end{bmatrix}
    if (src.startsWith('\\begin{', i)) {
      const envEnd = src.indexOf('}', i + 7);
      if (envEnd > i + 7) {
        const envName = src.slice(i + 7, envEnd).trim();
        const endTag = `\\end{${envName}}`;
        const closeIdx = src.indexOf(endTag, envEnd + 1);
        if (closeIdx > envEnd) {
          const inner = src.slice(envEnd + 1, closeIdx).trim();
          const rowStrs = inner.split(/\\\\|\n/).map(r => r.trim()).filter(Boolean);
          const rows = rowStrs.map(row =>
            row.split('&').map(cell => parseFormulaNodes(cell.trim())),
          );
          const [left, right] = MATRIX_DELIMS[envName] ?? ['[', ']'];
          nodes.push({ kind: 'matrix', left, right, rows });
          i = closeIdx + endTag.length;
          continue;
        }
      }
    }

    // 反斜杠命令
    if (ch === '\\') {
      const m = src.slice(i).match(/^\\([A-Za-z]+)/);
      if (m) {
        const cmd = m[1];
        i += m[0].length;

        if (cmd === 'frac' || cmd === 'dfrac' || cmd === 'tfrac') {
          const g1 = readBracedGroup(src, i);
          const g2 = g1 ? readBracedGroup(src, g1.nextIdx) : null;
          if (g1 && g2) {
            nodes.push({
              kind: 'frac',
              num: parseFormulaNodes(g1.content),
              den: parseFormulaNodes(g2.content),
            });
            i = g2.nextIdx;
            continue;
          }
        }

        if (cmd === 'sqrt') {
          let idxSkip = i;
          while (idxSkip < src.length && /\s/.test(src[idxSkip])) idxSkip += 1;
          if (src[idxSkip] === '[') {
            const closeBracket = src.indexOf(']', idxSkip);
            if (closeBracket > idxSkip) idxSkip = closeBracket + 1;
          }
          const body = readBracedGroup(src, idxSkip);
          if (body) {
            nodes.push({ kind: 'sqrt', body: parseFormulaNodes(body.content) });
            i = body.nextIdx;
            continue;
          }
        }

        if (cmd === 'bar' || cmd === 'vec' || cmd === 'hat' || cmd === 'overline') {
          const g = readBracedGroup(src, i);
          if (g) {
            const accent = cmd === 'vec' ? '⃗' : cmd === 'hat' ? '^' : '̄';
            const inner = latexToReadableMath(g.content);
            pushText(inner.length === 1 ? inner + accent : `${cmd}(${inner})`);
            i = g.nextIdx;
            continue;
          }
        }

        if (cmd === 'int' || cmd === 'iint' || cmd === 'oint' || cmd === 'sum' || cmd === 'prod' || cmd === 'lim') {
          const opSym = LATEX_SYMBOL_MAP[cmd] ?? cmd;
          let subNodes: MathNode[] | undefined;
          let supNodes: MathNode[] | undefined;
          let scan = i;
          for (let pass = 0; pass < 2; pass += 1) {
            while (scan < src.length && /\s/.test(src[scan])) scan += 1;
            if (src[scan] === '_' && !subNodes) {
              const g = readBracedGroup(src, scan + 1);
              if (g) {
                subNodes = parseFormulaNodes(g.content);
                scan = g.nextIdx;
              }
            } else if (src[scan] === '^' && !supNodes) {
              const g = readBracedGroup(src, scan + 1);
              if (g) {
                supNodes = parseFormulaNodes(g.content);
                scan = g.nextIdx;
              }
            }
          }
          i = scan;
          nodes.push({ kind: 'bigop', op: opSym, sub: subNodes, sup: supNodes });
          continue;
        }

        const sym = LATEX_SYMBOL_MAP[cmd];
        if (sym !== undefined) {
          const needSpace = /^[A-Za-z]/.test(sym) && i < src.length && /[A-Za-z0-9]/.test(src[i]);
          pushText(needSpace ? `${sym} ` : sym);
        } else {
          pushText(cmd);
        }
        continue;
      }
      if (i + 1 < src.length) {
        pushText(src[i + 1]);
        i += 2;
        continue;
      }
    }

    // 上下标 _ 与 ^
    if ((ch === '_' || ch === '^') && nodes.length > 0) {
      let sub: MathNode[] | undefined;
      let sup: MathNode[] | undefined;
      let scan = i;
      for (let pass = 0; pass < 2; pass += 1) {
        while (scan < src.length && /\s/.test(src[scan])) scan += 1;
        if (src[scan] === '_' && !sub) {
          const g = readBracedGroup(src, scan + 1);
          if (g) {
            sub = parseFormulaNodes(g.content);
            scan = g.nextIdx;
          }
        } else if (src[scan] === '^' && !sup) {
          const g = readBracedGroup(src, scan + 1);
          if (g) {
            sup = parseFormulaNodes(g.content);
            scan = g.nextIdx;
          }
        }
      }
      if (sub || sup) {
        const baseNode = nodes.pop()!;
        if (sub && sup) nodes.push({ kind: 'subsup', base: [baseNode], sub, sup });
        else if (sup) nodes.push({ kind: 'sup', base: [baseNode], sup });
        else if (sub) nodes.push({ kind: 'sub', base: [baseNode], sub });
        i = scan;
        continue;
      }
    }

    if (ch === '{' || ch === '}') {
      i += 1;
      continue;
    }

    pushText(ch);
    i += 1;
  }

  return nodes;
}

function nodesToReadableString(nodes: MathNode[]): string {
  return nodes
    .map((n) => {
      if (n.kind === 'text') return n.value;
      if (n.kind === 'frac') {
        const num = nodesToReadableString(n.num).trim();
        const den = nodesToReadableString(n.den).trim();
        const simpleNum = /^[A-Za-z0-9α-ωπ]+$/.test(num) || (!/[+\-*/=]/.test(num) && num.length <= 6) ? num : `(${num})`;
        const simpleDen = /^[A-Za-z0-9α-ωπ]+$/.test(den) || (!/[+\-*/=]/.test(den) && den.length <= 4) ? den : `(${den})`;
        return `${simpleNum}/${simpleDen}`;
      }
      if (n.kind === 'sqrt') {
        return `√(${nodesToReadableString(n.body).trim()})`;
      }
      if (n.kind === 'sup') {
        const base = nodesToReadableString(n.base);
        const supRaw = nodesToReadableString(n.sup).trim();
        const uni = toScript(supRaw, SUP_MAP);
        return uni ? `${base}${uni}` : `${base}^(${supRaw})`;
      }
      if (n.kind === 'sub') {
        const base = nodesToReadableString(n.base);
        const subRaw = nodesToReadableString(n.sub).trim();
        const uni = toScript(subRaw, SUB_MAP);
        return uni ? `${base}${uni}` : `${base}_${subRaw}`;
      }
      if (n.kind === 'subsup') {
        const base = nodesToReadableString(n.base);
        const subRaw = nodesToReadableString(n.sub).trim();
        const supRaw = nodesToReadableString(n.sup).trim();
        const uSub = toScript(subRaw, SUB_MAP) ?? `_${subRaw}`;
        const uSup = toScript(supRaw, SUP_MAP) ?? `^(${supRaw})`;
        return `${base}${uSub}${uSup}`;
      }
      if (n.kind === 'bigop') {
        const sub = n.sub ? nodesToReadableString(n.sub).trim() : '';
        const sup = n.sup ? nodesToReadableString(n.sup).trim() : '';
        if (n.op === 'lim') {
          return sub ? `lim(${sub}) ` : 'lim ';
        }
        const uSub = sub ? (toScript(sub, SUB_MAP) ?? `_${sub}`) : '';
        const uSup = sup ? (toScript(sup, SUP_MAP) ?? `^(${sup})`) : '';
        return `${n.op}${uSub}${uSup} `;
      }
      if (n.kind === 'matrix') {
        const rowTxt = n.rows
          .map((r) =>
            r
              .map((c) => nodesToReadableString(c).trim().replace(/[,，]+$/, ''))
              .filter(Boolean)
              .join(', '),
          )
          .join(' ; ');
        return `${n.left}${rowTxt}${n.right}`;
      }
      return '';
    })
    .join('');
}

/** 切分同时含 $$...$$、$$...$$ / $...$ / \(...\) / \[...\] 的混合文本为结构化段落 */
export function parseMathSegments(rawText: string): RichSegment[] {
  const segments: RichSegment[] = [];
  const regex = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\$[^$\n]+\$|\\\([\s\S]+?\\\))/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(rawText)) !== null) {
    if (match.index > lastIndex) {
      const plain = rawText.slice(lastIndex, match.index);
      if (plain.trim()) segments.push({ type: 'text', raw: plain });
    }
    const token = match[0];
    if (token.startsWith('$$') && token.endsWith('$$')) {
      segments.push({ type: 'display_math', raw: token.slice(2, -2).trim() });
    } else if (token.startsWith('\\[') && token.endsWith('\\]')) {
      segments.push({ type: 'display_math', raw: token.slice(2, -2).trim() });
    } else if (token.startsWith('\\(') && token.endsWith('\\)')) {
      segments.push({ type: 'inline_math', raw: token.slice(2, -2).trim() });
    } else {
      segments.push({ type: 'inline_math', raw: token.slice(1, -1).trim() });
    }
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < rawText.length) {
    const tail = rawText.slice(lastIndex);
    if (tail.trim()) segments.push({ type: 'text', raw: tail });
  }
  return segments;
}

/**
 * 将混合文本中的所有 LaTeX 片段（$...$、\(...\)、\[...\] 或裸 LaTeX 命令）
 * 转换为紧凑、高可读性的 Unicode 数学文本，绝不残留未闭合的反斜杠/括号碎片。
 */
export function latexToReadableMath(input: string): string {
  const trimmed = input.trim();
  if (!trimmed) return '';

  const segments = parseMathSegments(trimmed);
  if (segments.length > 0 && segments.some((s) => s.type !== 'text')) {
    return segments
      .map((s) => {
        if (s.type !== 'text') {
          return nodesToReadableString(parseFormulaNodes(s.raw)).replace(/\s{2,}/g, ' ').trim();
        }
        return /\\[A-Za-z]+|[_^]\{/.test(s.raw)
          ? nodesToReadableString(parseFormulaNodes(s.raw)).replace(/\s{2,}/g, ' ').trim()
          : s.raw;
      })
      .join(' ')
      .replace(/\s+([，。；：、？！,.])/g, '$1')
      .replace(/\s{2,}/g, ' ')
      .trim();
  }

  // 若整串含裸 LaTeX 命令/上下标（如 c_{11} 或 \int_{0}^{x}），同样转写为易读式子
  if (/\\[A-Za-z]+|[_^]\{/.test(trimmed)) {
    return nodesToReadableString(parseFormulaNodes(trimmed))
      .replace(/\s{2,}/g, ' ')
      .trim();
  }
  return trimmed;
}
