import {
  filterSafeHintText,
  PREWARM_TIMEOUT_MS,
  resolveChatEndpoint,
  type CompletionResponse,
  type HintLevel,
} from '../hints/safety';
import type { StallType } from '../stalls';
import { latexToReadableMath } from '../utils/mathFormat';
import {
  HINT_STYLES,
  HINT_STYLE_LABELS,
  type HintStyle,
} from './personalizedPolicy';

export interface SolutionStepSpec {
  stepIndex: number;
  goalDescription: string;
  expectedTokens: string[];
}

export interface SolutionPitfallSpec {
  triggerPattern: string;
  warningHint: string;
}

export interface StyleHintTiers {
  l0?: string;
  l1?: string;
  l2?: string;
  l3?: string;
}

export interface SolutionCacheEntry {
  problemKey: string;
  problemText: string;
  steps: SolutionStepSpec[];
  commonPitfalls: SolutionPitfallSpec[];
  hintsByStyle: Partial<Record<HintStyle, StyleHintTiers>>;
  /** 密封的最终答案占位（仅预留给内部防泄漏与后续核对，严禁展示给学生） */
  sealedFinalAnswer?: string;
  sourceMode: 'cloud_prewarm' | 'cloud_live' | 'edge_offline';
  updatedAt: number;
}

export function normalizeProblemKey(text: string): string {
  return text.trim().replace(/\s+/g, ' ');
}

/** 大学数学高频结构线索与易错陷阱模式（供本地图谱蒸馏与端侧合成共用） */
export const MATH_DOMAIN_KNOWLEDGE: Array<{
  pattern: RegExp;
  clue: string;
  triggerPattern: string;
  warningHint: string;
}> = [
  {
    pattern: /\\?lim\b|极限|泰勒|洛必达|无穷小|未定式|L'H[oô]pital|squeeze theorem/i,
    clue: '分母主部等价阶数与“上下同阶”泰勒展开原则',
    triggerPattern: '截断阶数不够，或加减项中误用局部等价无穷小',
    warningHint: '加减运算里不能直接把部分项做等价无穷小替换；先看看分母最低非零幂次是几阶？',
  },
  {
    pattern: /∫|\\i{1,3}nt|\\oint|变上限|二重积分|积分|分部积分|polar coordinates|double integral/i,
    clue: '变上限求导/分部积分选取或二重积分区域定限与极坐标雅可比 r',
    triggerPattern: '变限积分求导漏换变量，或极坐标二重积分漏乘雅可比因子 r',
    warningHint: '核对对变限积分求导后的代换、分部积分符号，以及极坐标变换有没有漏乘面积元因子 r？',
  },
  {
    pattern: /微分方程|y''|y'|特解|\bODE\b|differential equation|characteristic equation|齐次|通解/i,
    clue: '齐次特征方程重根判定与特解待定形式 / 一阶线性积分因子',
    triggerPattern: '特解形式未根据特征根重数乘 x^k，或一阶线性微分方程漏乘积分因子',
    warningHint: '检查自由项里的指数是否与特征根重合（重合需补乘 x^k），以及首项系数和常数 C 是否处理妥当。',
  },
  {
    pattern: /矩阵|行列式|秩|Ax\s*=|方程组|\\det|\|A\|/i,
    clue: '方阵系数行列式 |A|=0 临界点及增广矩阵行秩比对',
    triggerPattern: '未先讨论 |A|=0 漏掉临界零点，或直接带参数行变换误除含参因式',
    warningHint: '系数矩阵含参数时，别直接同除可能是 0 的表达式！先算出行列式 |A|=0 锁定临界点会更稳。',
  },
  {
    pattern: /特征值|正交|对称矩阵|相似对角化/i,
    clue: '实对称矩阵不同特征值向量正交与重根施密特正交化',
    triggerPattern: '多重特征向量忘记先施密特正交化直接单位化',
    warningHint: '重特征值求出的两个线性无关特征向量之间天然正交吗？组装正交矩阵 Q 前别忘了先正交再单位化。',
  },
  {
    pattern: /贝叶斯|全概率|条件概率|\bP\(|随机变量|正态|\bPhi\b|Φ|期望|方差|均匀分布/i,
    clue: '完备事件组先验加权、条件概率竖杠顺序与标准化公式',
    triggerPattern: '全概率分母漏乘各分支先验权重，或正态标准化误除以方差而非标准差',
    warningHint: '检查两处高发点：条件概率竖杠前后有没有写反？全概率/标准化时先验权重和标准差有没有算准？',
  },
];

/**
 * 本地离线多风格四层提示库（涵盖 L0 微轻推 与 6 种大学数学启发风格）。
 */
export function getOfflineStyledHint(
  type: StallType,
  level: HintLevel,
  style: HintStyle = 'socratic_question',
): string {
  if (level === 0) {
    if (style === 'algebra_check') return '微提醒：留意一下这一步的正负号与系数。';
    if (style === 'sanity_check') return '微提醒：代个特值或零点快速心算检验一下。';
    if (style === 'condition_reminder') return '微提醒：检查题干约束边界用全了吗？';
    return '微提醒：稳住当前节奏，小步顺推即可。';
  }

  if (style === 'socratic_question') {
    if (level === 3) return '还是没思路的话，先把这一步的中间目标写成已知等式，再观察还能消去哪个变量。';
    if (level === 2) return '试着把当前这一步拆成更小的目标，一步一步来。';
    if (type === 'pause_erase') {
      return '这里卡住了吗？试试把题目的条件再读一遍，看看有没有没用上的。';
    }
    if (type === 'speed_drop') {
      return '思路断了吗？回想一下上一步用到了什么结论。';
    }
    return '思考遇阻了吗？先问自己：这一步最终想消掉或求出哪个关键量？';
  }

  if (style === 'condition_reminder') {
    if (level === 1) return '先停笔扫一遍题干，看看哪个已知条件或取值范围还没用上？';
    if (level === 2) return '把题干中的已知关系逐条列出来，检查还缺哪一环即可衔接。';
    return '先把用到的定理前提和题目给定条件逐一对齐，再确定下一步消元方向。';
  }

  if (style === 'algebra_check') {
    if (level === 1) return '刚才那一步变号或移项有没有小疏漏？回头核对一下系数与符号。';
    if (level === 2) return '试着代一个简单特值检验当前算式，确认等价变形没有算偏。';
    return '检查高次项系数是否为零、分母与定义域边界是否遗漏讨论，确认无误再往下推。';
  }

  if (style === 'concept_recall') {
    if (level === 1) return '先别急着硬算，回忆这一章对应定理成立的核心前提是什么？';
    if (level === 2) return '想想高数/线代/概率中联系题设与目标的那条核心恒等式或正交性质。';
    return '把题干特征与教科书标准定理对齐：先确定方法模型（如泰勒展开/增广秩/贝叶斯），再动手变形。';
  }

  if (style === 'sanity_check') {
    if (level === 1) return '暂停检查一下当前表达式的极限趋势或特殊值（如取 0 或 1），结果合理吗？';
    if (level === 2) return '对照一下行列式零点或概率总和归一项，看看刚才的变形有没有漏项或反号。';
    return '利用临界点检验、奇偶对称性或矩阵正交内积快速自查上一步推导后再继续。';
  }

  if (level === 1) return '这一步跨度有些大，能不能先在中间插一个更小的过渡目标？';
  if (level === 2) return '试着把当前这一步拆成更小的目标，先写最确定的一半。';
  return '把大问题拆成“先求中间量 → 再代回原式”两小段，先只完成第一段推导。';
}

const LATEX_ESCAPE_COLLISIONS: Record<string, RegExp> = {
  b: /^(?:ar|eta|inom|egin|ig|matrix|old|mod)/,
  f: /^(?:rac|orall|lat|box)/,
  n: /^(?:abla|eq|ot|u\b|i\b)/,
  r: /^(?:ho|ight)/,
  t: /^(?:o\b|heta|imes|au|an|ext|frac|riangle)/,
};

/**
 * 修复数学模型输出 JSON 字符串内未双反斜杠转义的 LaTeX 命令（如 \frac, \to, \neq, \int）。
 */
export function repairJsonLatexEscapes(raw: string): string {
  let out = '';
  let inString = false;
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (!inString) {
      if (ch === '"') inString = true;
      out += ch;
      continue;
    }
    if (ch === '"') {
      inString = false;
      out += ch;
      continue;
    }
    if (ch === '\\') {
      const next = raw[i + 1];
      if (next === undefined) {
        out += '\\\\';
        continue;
      }
      if (next === '"' || next === '\\' || next === '/') {
        out += ch + next;
        i += 1;
        continue;
      }
      if (next === 'u' && /^[0-9a-fA-F]{4}$/.test(raw.slice(i + 2, i + 6))) {
        out += raw.slice(i, i + 6);
        i += 5;
        continue;
      }
      const collisionRe = LATEX_ESCAPE_COLLISIONS[next];
      if (collisionRe) {
        // 若匹配 \frac, \to, \theta, \nabla, \neq, \rho, \bar, \beta 等 LaTeX 命令，补成双反斜杠防被当成控制符吞掉
        if (collisionRe.test(raw.slice(i + 2))) {
          out += '\\\\';
          continue;
        }
        out += ch + next;
        i += 1;
        continue;
      }
      // 其余 \int, \sum, \lambda, \{ 等一律自动补成 \\ 以通过 JSON.parse
      out += '\\\\';
      continue;
    }
    if (ch === '\n' || ch === '\r') {
      out += ch === '\n' ? '\\n' : '\\r';
      continue;
    }
    out += ch;
  }
  return out.replace(/,\s*([}\]])/g, '$1');
}

function tryParseLooseJson(rawSlice: string): unknown {
  for (const cand of [repairJsonLatexEscapes(rawSlice), rawSlice]) {
    try {
      return JSON.parse(cand);
    } catch {
      // ignore
    }
  }
  return null;
}

function extractArrayField(text: string, fieldName: string): unknown[] | undefined {
  const m = text.match(new RegExp(`"${fieldName}"\\s*:\\s*\\[`));
  if (!m || m.index === undefined) return undefined;
  const openIdx = text.indexOf('[', m.index);
  let depth = 0;
  let inStr = false;
  for (let i = openIdx; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"' && text[i - 1] !== '\\') inStr = !inStr;
    if (inStr) continue;
    if (ch === '[') depth += 1;
    else if (ch === ']') {
      depth -= 1;
      if (depth === 0) {
        const arr = tryParseLooseJson(text.slice(openIdx, i + 1));
        return Array.isArray(arr) ? arr : undefined;
      }
    }
  }
  return undefined;
}

export function parseSolutionGraphJson(raw: string): Partial<SolutionCacheEntry> | null {
  const noThink = raw.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  const stripped = noThink
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/```\s*$/i, '')
    .trim();
  const start = stripped.indexOf('{');
  const end = stripped.lastIndexOf('}');
  if (start >= 0 && end > start) {
    const parsed = tryParseLooseJson(stripped.slice(start, end + 1));
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Partial<SolutionCacheEntry>;
    }
  }

  // 若尾部 hintsByStyle 被截断，尽量抢救前半段完整的 steps / commonPitfalls / sealedFinalAnswer
  if (start >= 0) {
    const tailSlice = stripped.slice(start);
    const steps = extractArrayField(tailSlice, 'steps') as SolutionStepSpec[] | undefined;
    const pitfalls = extractArrayField(tailSlice, 'commonPitfalls') as SolutionPitfallSpec[] | undefined;
    const sfaMatch = tailSlice.match(/"sealedFinalAnswer"\s*:\s*("[^"]*"|\{[^}]*\})/);
    const sfaVal = sfaMatch ? tryParseLooseJson(sfaMatch[1]) : undefined;
    if (steps && steps.length > 0) {
      return {
        steps,
        ...(pitfalls ? { commonPitfalls: pitfalls } : {}),
        ...(sfaVal !== undefined ? { sealedFinalAnswer: sfaVal as string } : {}),
      };
    }
  }
  return null;
}

/** 从带平衡大括号的 \boxed{...} 中完整提取封存答案 */
export function extractBoxedAnswer(text: string): string | undefined {
  const idx = text.lastIndexOf('\\boxed{');
  if (idx >= 0) {
    let depth = 0;
    let bodyStart = -1;
    for (let i = idx + 6; i < text.length; i++) {
      const ch = text[i];
      if (ch === '{') {
        if (depth === 0) bodyStart = i + 1;
        depth += 1;
      } else if (ch === '}') {
        depth -= 1;
        if (depth === 0 && bodyStart >= 0) {
          return text.slice(bodyStart, i).trim();
        }
      }
    }
  }
  const zhMatch = text.match(/(?:最终答案|答案|结果为|解得)\s*[:：为是]?\s*([^\n。；;]{1,48})/);
  return zhMatch?.[1]?.trim();
}

function extractMathTokensFromText(text: string): string[] {
  const tokens = new Set<string>();
  const matches = text.match(
    /\\(?:frac|int|oint|sum|lim|to|matrix|det|sin|cos|ln|exp|sqrt|lambda|Phi|alpha|beta|bar)|\|A\||[=]/g,
  );
  if (matches) {
    for (const m of matches.slice(0, 6)) tokens.add(m);
  }
  if (tokens.size === 0) tokens.add('=');
  return Array.from(tokens);
}

const ECHO_PLACEHOLDER_RE =
  /^(?:\.{2,}|目标|第一步目标|第1步数学目标|温和反问|温和核对反问|典型误区|典型易错点|最终结果|关键表达式)$/i;

/** 过滤小模型照搬示例模板或过短的占位文本 */
function cleanModelText(raw: unknown, minLen = 6): string {
  if (typeof raw !== 'string') return '';
  const t = raw.replace(/^(?:目标|提示|思路)[:：]\s*/i, '').trim();
  if (t.length < minLen || ECHO_PLACEHOLDER_RE.test(t)) return '';
  return t;
}

const PREWARM_SYSTEM_MESSAGE = [
  '你是《笔迹脉搏》数学解题图谱分析师。你需要准确拆解题目核心推导步骤、易错歧路与分级启发提示。',
  '允许将最终结论单独封存在 sealedFinalAnswer 字段中（系统会自动加密封存不给学生看），',
  '但 steps[].goalDescription、commonPitfalls 与所有 hints 文本中严禁直接写出最终结果，只能启发下一步思路。',
].join('');

const EN_TO_ZH_STEP_RULES: Array<[RegExp, string]> = [
  [/^Understanding the Problem[:：]?.*$/i, '明确题干已知条件与待求目标关系'],
  [/^(?:Using|Apply) the Squeeze Theorem.*$/i, '构造上下界不等式并应用夹逼准则'],
  [/^Evaluating the Limits of bounding functions.*$/i, '分别计算左右/上下边界函数的极限'],
  [/^Applying the Squeeze Theorem.*$/i, '由两边边界极限相同推出中间式极限'],
  [/^To solve the given second-order.*?equation[:：]?\s*/i, '先求二阶齐次方程通解，再设非齐次特解：'],
  [/^(?:Step \d+[:：]\s*)?Solve the Homogeneous Equation[:：]?.*$/i, '第一步：求解对应齐次方程 y′′ + py′ + qy = 0'],
  [/^The homogeneous equation is[:：]?.*$/i, '先列出对应的齐次微分方程特征方程'],
  [/^(?:To solve this,?\s*)?(?:we\s+)?first find the characteristic equation[:：]?.*$/i, '写出并求解特征方程 r² + pr + q = 0'],
  [/^First,?\s*(?:let's\s+)?rewrite the inequality defining the region.*$/i, '先把积分区域 D 的不等式配成标准圆/边界方程'],
  [/^(?:Next|Then),?\s*we use polar coordinates to simplify.*$/i, '转为极坐标 x = r cosθ, y = r sinθ 并写对 dxdy = r drdθ'],
  [/^(?:First|Next),?\s*integrate with respect to ([A-Za-z]).*$/i, '先确定上下限并对变量 $1 积分'],
  [/^Next,?\s*use the identity .*$/i, '利用三角降幂或恒等式化简后继续外层积分'],
  [/^First,?\s*calculate\s*/i, '先求分段点极限/导数条件：'],
  [/^Next,?\s*check the differentiability condition.*$/i, '再检验分段点连续性与左右导数相等条件'],
  [/^First derivative for\s*/i, '计算区间内一阶导数：'],
  [/^First,?\s*(?:let's\s+)?perform (?:some )?row operations.*$/i, '对增广矩阵实施初等行变换化为阶梯形'],
  [/^(?:Next|Then),?\s*we eliminate .*$/i, '消去主元下方元素，讨论主对角线含参临界值'],
  [/^(?:Identify|Determine) the (?:form of the )?limit(?: expression)?[:：]?\s*/i, '判别极限未定式类型：'],
  [/^(?:Recall|Use) (?:the|a) (?:known|standard) (?:result|limit)(?: from calculus)?[:：]?.*$/i, '调用重要极限或常用等价基本公式'],
  [/^(?:State the final answer|Final answer|Evaluate the final limit)[:：]?.*$/i, '代入边界完成最终化简'],
  [/^To (?:solve|evaluate) the (?:definite |indefinite )?integral\s*/i, '分析被积函数并确定积分路线：'],
  [/^To find (?:the )?explicit (?:form|expression) of\s*/i, '梳理已知函数关系与目标表达式：'],
  [/^To find (?:the )?eigenvalues(?: and eigenvectors)? of(?: matrix)?\s*/i, '围绕矩阵构造特征多项式求解特征值：'],
  [/^Let(?:'s| us)? denote (?:the integral term )?by (?:a new function )?/i, '设辅助函数分离变量或积分项：'],
  [/^Define the functions?[:：]?\s*/i, '拆分构造辅助函数：'],
  [/^Differentiate both sides(?: of the equation)? with respect to ([A-Za-z])[,:]?\s*/i, '等式两边对 $1 求导并化为微分方程：'],
  [/^Apply L'H[oô]pital(?:'s)? Rule(?:\s+(?:again|one more time))?[:：]?\s*/i, '对分子分母应用洛必达法则求导：'],
  [/^The antiderivative of .+$/i, '先求出被积函数的一个原函数（注意符号与系数）'],
  [/^This means we need to evaluate .+$/i, '利用牛顿-莱布尼茨公式代入上下限区间端点'],
  [/^First,?\s*compute(?: the derivatives?)?[:：]?\s*/i, '首先计算导数或初等变换：'],
  [/^(?:Next|Then|Now),?\s*(?:we\s+)?(?:compute|find|solve|write down|rewrite)\s+/i, '接着整理推导关系：'],
  [/^Find the characteristic equation[:：]?\s*/i, '列写特征方程 det(A - λI) = 0：'],
  [/^The characteristic equation is(?: given by)?[:：]?\s*/i, '特征方程形式为：'],
  [/^Compute the determinant(?: of this matrix)?[:：]?\s*/i, '展开特征行列式计算：'],
  [/^Solve the (?:quadratic |characteristic )?equation[:：]?\s*/i, '解出特征方程并检验根的重数：'],
  [/^Integrate both sides(?: with respect to [A-Za-z])?[,:]?\s*/i, '两边积分（留意积分因子或常数 C）：'],
  [/^Use integration by parts.*$/i, '按“反对幂指三”确定 u 与 dv 做分部积分'],
  [/^Use substitution.*$/i, '选取合适变量代换化简被积式'],
  [/^(?:Finally|Conclusion)[,:：]?\s*/i, '最后代入边界或初值条件检验收敛/结果'],
  [/^where\s+(.+?)\s+is the identity matrix.*$/i, '其中 $1 为单位矩阵'],
];

function localizeMathStepGoal(raw: string): string {
  let out = raw.trim();
  for (const [re, zh] of EN_TO_ZH_STEP_RULES) {
    if (re.test(out)) {
      out = out.replace(re, zh).trim();
    }
  }
  out = latexToReadableMath(out)
    .replace(/\\(?:\(|\)|\[|\])/g, '')
    .replace(/[：:]\s*$/, '')
    .trim();
  return out;
}

function smartTruncateGoal(text: string, maxLen = 52): string {
  if (text.length <= maxLen) return text;
  let cut = text.slice(0, maxLen).trim();
  // 避免在左括号内部被粗暴截断
  const lastOpenParen = Math.max(cut.lastIndexOf('('), cut.lastIndexOf('（'), cut.lastIndexOf('['));
  const lastCloseParen = Math.max(cut.lastIndexOf(')'), cut.lastIndexOf('）'), cut.lastIndexOf(']'));
  if (lastOpenParen > lastCloseParen && lastOpenParen >= 14) {
    cut = cut.slice(0, lastOpenParen).trim();
  }
  return cut.replace(/[:：,，(（\[\\]+$/, '').trim() + '…';
}

function stripSpoilerFromGoal(rawLine: string, sealed?: string): string {
  let cleaned = rawLine
    .replace(/^[-*•]\s*/g, '')
    .replace(/^[\d一二三四五六七八九十]+[.、)\]]\s*/g, '')
    .replace(/^(?:goalDescription|步骤目标|第[一二三四五六七八九十\d]+步|Step\s*\d+)\s*[:：.\-]?\s*/i, '')
    .replace(/^#+\s*/g, '')
    // 仅清理 Markdown 粗体/行内代码标记，绝不误删 LaTeX 下标 '_'（如 c_{11}、\int_{0}^{x}）
    .replace(/[*`]/g, '')
    .replace(/(^|\s)_([^_\n]+)_(\s|$)/g, '$1$2$3')
    .replace(/\\boxed\{[\s\S]*?\}/g, '')
    .replace(/(?:因此|故|所以|从而|得|解得)\s*(?:答案|最终结果|原式|极限|积分)?\s*[=＝为是].*$/g, '')
    .replace(/=\s*[-+]?\\?(?:cos|sin|ln|exp|arctan|frac)[^,;。；\n]*\\Big\|.*$/i, '（代入上下限计算）')
    .replace(/\s+/g, ' ')
    .trim();

  if (sealed && sealed.length >= 2) {
    cleaned = cleaned.split(sealed).join('待求量').trim();
  }

  // 若整步变成裸“c_{11}”等纯矩阵元索引碎片，则跳过无意义步骤
  if (/^(?:\\\(\s*)?[A-Za-z]_?\{\d+\}(?:\s*\\\))?[:：]?$/i.test(cleaned)) {
    return '';
  }

  const localized = localizeMathStepGoal(cleaned);
  return smartTruncateGoal(localized, 56);
}

/** 从步骤图谱与易错歧路补全全部 6 种启发风格的 L0-L3 分级提示，并严防封存答案泄漏 */
export function buildCompleteHintsFromGraph(
  problemText: string,
  steps: SolutionStepSpec[],
  commonPitfalls: SolutionPitfallSpec[],
  rawStyles: Partial<Record<HintStyle, StyleHintTiers>> | undefined,
  sealedFinalAnswer?: string,
): Partial<Record<HintStyle, StyleHintTiers>> {
  const step0 = steps[0]?.goalDescription;
  const step1 = steps[1]?.goalDescription ?? step0;
  const step2 = steps[Math.min(2, Math.max(0, steps.length - 1))]?.goalDescription ?? step1;
  const pitfall0 = commonPitfalls[0]?.warningHint;
  const domainMatch = MATH_DOMAIN_KNOWLEDGE.find(k => k.pattern.test(problemText));
  const clue = domainMatch?.clue ?? '题设给定条件与等价结构关系';

  const result: Partial<Record<HintStyle, StyleHintTiers>> = {};

  for (const style of HINT_STYLES) {
    const incoming = rawStyles?.[style];
    let l0 = filterSafeHintText(cleanModelText(incoming?.l0, 4), sealedFinalAnswer);
    let l1 = filterSafeHintText(cleanModelText(incoming?.l1, 8), sealedFinalAnswer);
    let l2 = filterSafeHintText(cleanModelText(incoming?.l2, 8), sealedFinalAnswer);
    let l3 = filterSafeHintText(cleanModelText(incoming?.l3, 8), sealedFinalAnswer);

    if (!l0) {
      if (step0) {
        const shortStep = step0.slice(0, 14);
        l0 =
          style === 'algebra_check'
            ? `微核对：留意「${shortStep}」正负号`
            : `微提醒：关注「${shortStep}」`;
      } else {
        l0 = getOfflineStyledHint('cognitive_stall', 0, style);
      }
    }

    if (!l1) {
      if (style === 'condition_reminder') {
        l1 = step0
          ? `先读题干里的关键条件，试着围绕「${step0}」找切入点？`
          : `结合题设与「${clue}」，先确认哪条约束还没用上？`;
      } else if (style === 'algebra_check') {
        l1 =
          pitfall0 ||
          (step1 ? `核对「${step1}」处的符号、系数与定义域边界有没有算偏？` : getOfflineStyledHint('pause_erase', 1, style));
      } else if (style === 'step_decompose') {
        l1 = step0 ? `先把这一段拆小：第一步先只完成「${step0}」。` : getOfflineStyledHint('cognitive_stall', 1, style);
      } else if (style === 'concept_recall') {
        l1 = `回顾一下与「${clue}」关联的核心定理或前提条件是什么？`;
      } else if (style === 'sanity_check') {
        l1 = '暂停代入特值（如 0 或 1）或检查极限趋势/对称性，确认当前变形合理吗？';
      } else {
        l1 = step0 ? `推进到这里了，围绕「${step0}」想一想下一步用哪条性质？` : getOfflineStyledHint('pause_erase', 1, style);
      }
    }

    if (!l2) {
      if (style === 'step_decompose' && step0 && step1) {
        l2 = `拆成两小步：①先做「${step0}」；②再推进「${step1}」。`;
      } else if (step1) {
        l2 = `完成起手后，下一步可动手尝试「${step1}」。`;
      } else {
        l2 = getOfflineStyledHint('cognitive_stall', 2, style);
      }
    }

    if (!l3) {
      if (steps.length >= 2) {
        const summary = steps
          .slice(0, 3)
          .map((s, idx) => `第${idx + 1}步（${latexToReadableMath(s.goalDescription)}）`)
          .join(' → ');
        l3 = `主干路线：${summary}，其余演算留给你亲手收尾。`;
      } else {
        l3 = getOfflineStyledHint('cognitive_stall', 3, style);
      }
    }

    result[style] = {
      l0: filterSafeHintText(latexToReadableMath(l0), sealedFinalAnswer) ?? getOfflineStyledHint('cognitive_stall', 0, style),
      l1: filterSafeHintText(latexToReadableMath(l1), sealedFinalAnswer) ?? getOfflineStyledHint('cognitive_stall', 1, style),
      l2: filterSafeHintText(latexToReadableMath(l2), sealedFinalAnswer) ?? getOfflineStyledHint('cognitive_stall', 2, style),
      l3: filterSafeHintText(latexToReadableMath(l3), sealedFinalAnswer) ?? getOfflineStyledHint('cognitive_stall', 3, style),
    };
  }

  return result;
}

/**
 * 针对本地数学强化小模型（Qwen2-Math / DeepSeek-R1-Distill-Qwen）的 CoT 蒸馏器：
 * 当小模型优先按照数学思维链逐行推导出步骤与 \boxed{answer} 而非严格 JSON 时，
 * 自动解析出主干步骤 steps、封存答案 sealedFinalAnswer、匹配易错歧路并生成全部 6 风格 L0-L3 提示。
 */
export function distillSolutionGraphFromMathCoT(
  problemText: string,
  rawResponse: string,
  sourceMode: SolutionCacheEntry['sourceMode'] = 'edge_offline',
): SolutionCacheEntry | null {
  const body = rawResponse.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  if (!body) return null;

  const sfaInline = body.match(/sealedFinalAnswer\s*[:：]\s*([^\n]{1,64})/i)?.[1]?.trim();
  const sealedFinalAnswer =
    extractBoxedAnswer(rawResponse) ??
    (sfaInline && !ECHO_PLACEHOLDER_RE.test(sfaInline) ? sfaInline : undefined);

  const SCHEMA_KEY_LINE_RE =
    /^[-*•]?\s*(?:steps|commonPitfalls|expectedTokens|sealedFinalAnswer|hintsByStyle|socratic_question|algebra_check|triggerPattern|warningHint|stepIndex)\s*[:：]/i;
  const rawLines = body
    .split(/\n+/)
    .map(l => l.trim())
    .filter(
      l =>
        l.length > 0 &&
        !/^\\?[\[\]]$/.test(l) &&
        !SCHEMA_KEY_LINE_RE.test(l) &&
        !/^(?:the\s+(?:final\s+)?answer\s+is|therefore,?\s+the\s+solution\s+is|state\s+the\s+final\s+answer)/i.test(l),
    );

  const steps: SolutionStepSpec[] = [];
  const prioritizedLines = rawLines.filter(l =>
    /^(?:\d+[.)、]|Step\s*\d+|第[一二三四五六七八九\d]+步|[-*]\s*goalDescription|First|Next|Finally|首先|其次|然后|最后)/i.test(
      l,
    ),
  );
  const scanLines = prioritizedLines.length >= 2 ? prioritizedLines : rawLines;

  for (const line of scanLines) {
    if (/\\boxed\{/.test(line) && line.length < 36) continue;
    const goal = stripSpoilerFromGoal(line, sealedFinalAnswer);
    if (goal.length < 6 || ECHO_PLACEHOLDER_RE.test(goal)) continue;
    if (!filterSafeHintText(goal, sealedFinalAnswer)) continue;
    steps.push({
      stepIndex: steps.length,
      goalDescription: goal,
      expectedTokens: extractMathTokensFromText(line),
    });
    if (steps.length >= 5) break;
  }

  // 若整段文本过短未抽出多步，则按题干数学线索保底构建起步里程碑（优先匹配题干正文领域）
  const problemMatches = MATH_DOMAIN_KNOWLEDGE.filter(k => k.pattern.test(problemText));
  const bodyMatches = MATH_DOMAIN_KNOWLEDGE.filter(
    k => !k.pattern.test(problemText) && k.pattern.test(body),
  );
  const domainMatches = [...problemMatches, ...bodyMatches];
  if (steps.length === 0) {
    const clue = domainMatches[0]?.clue ?? '整理题设已知式与目标关系';
    steps.push({
      stepIndex: 0,
      goalDescription: `识别题型关键线索：围绕「${clue}」建立起手关系`,
      expectedTokens: extractMathTokensFromText(problemText + ' ' + body),
    });
  }

  const commonPitfalls: SolutionPitfallSpec[] = domainMatches.slice(0, 2).map(m => ({
    triggerPattern: m.triggerPattern,
    warningHint: m.warningHint,
  }));

  const hintsByStyle = buildCompleteHintsFromGraph(
    problemText,
    steps,
    commonPitfalls,
    undefined,
    sealedFinalAnswer,
  );

  return {
    problemKey: normalizeProblemKey(problemText),
    problemText: problemText.trim(),
    steps,
    commonPitfalls,
    hintsByStyle,
    ...(sealedFinalAnswer ? { sealedFinalAnswer } : {}),
    sourceMode,
    updatedAt: Date.now(),
  };
}

/**
 * 解题图谱缓存器（SolutionCache）：
 * 在学生审题 5~30 秒内静默调用本地数学强化小模型或云端模型预解步骤图谱、易错歧路与 6 风格 L0-L3 提示，并封存答案。
 */
export class SolutionCacheManager {
  private entries = new Map<string, SolutionCacheEntry>();
  private inflightPrewarm = new Map<string, Promise<SolutionCacheEntry | null>>();

  public hasPrewarmed(problemText: string): boolean {
    const key = normalizeProblemKey(problemText);
    if (!key) return false;
    const entry = this.entries.get(key);
    return Boolean(entry && Object.keys(entry.hintsByStyle).length > 0);
  }

  public getEntry(problemText: string): SolutionCacheEntry | null {
    const key = normalizeProblemKey(problemText);
    if (!key) return null;
    return this.entries.get(key) ?? null;
  }

  public preloadEntry(entry: SolutionCacheEntry): void {
    const key = normalizeProblemKey(entry.problemText || entry.problemKey);
    if (!key) return;
    if (!this.entries.has(key)) {
      this.entries.set(key, { ...entry, problemKey: key });
    }
  }

  public hydrateEntries(entries: SolutionCacheEntry[]): void {
    for (const entry of entries) {
      if (entry && typeof entry.problemText === 'string') {
        this.preloadEntry(entry);
      }
    }
  }

  public lookupHint(
    problemText: string,
    level: HintLevel,
    style: HintStyle,
  ): { text: string; sourceMode: SolutionCacheEntry['sourceMode'] } | null {
    const key = normalizeProblemKey(problemText);
    if (!key) return null;
    const entry = this.entries.get(key);
    if (!entry) return null;

    const tiers = entry.hintsByStyle[style] ?? entry.hintsByStyle.socratic_question;
    if (!tiers) return null;
    const text = level === 0 ? tiers.l0 : level === 1 ? tiers.l1 : level === 2 ? tiers.l2 : tiers.l3;
    const safe = filterSafeHintText(text, entry.sealedFinalAnswer);
    if (!safe) return null;
    return { text: safe, sourceMode: entry.sourceMode };
  }

  public storeHint(
    problemText: string,
    level: HintLevel,
    style: HintStyle,
    hintText: string,
    sourceMode: SolutionCacheEntry['sourceMode'] = 'cloud_live',
  ): void {
    const key = normalizeProblemKey(problemText);
    if (!key) return;
    const existing = this.entries.get(key) ?? {
      problemKey: key,
      problemText: problemText.trim(),
      steps: [],
      commonPitfalls: [],
      hintsByStyle: {},
      sourceMode,
      updatedAt: Date.now(),
    };
    const safeText = filterSafeHintText(hintText, existing.sealedFinalAnswer);
    if (!safeText) return;

    const tiers = existing.hintsByStyle[style] ?? {};
    if (level === 0) tiers.l0 = safeText;
    else if (level === 1) tiers.l1 = safeText;
    else if (level === 2) tiers.l2 = safeText;
    else tiers.l3 = safeText;
    existing.hintsByStyle[style] = tiers;
    existing.updatedAt = Date.now();
    this.entries.set(key, existing);
  }

  public async prewarm(problemText: string): Promise<SolutionCacheEntry | null> {
    const key = normalizeProblemKey(problemText);
    if (!key) return null;
    if (this.hasPrewarmed(key)) return this.entries.get(key) ?? null;

    const edgeBase = process.env.EXPO_PUBLIC_EDGE_LLM_URL?.trim().replace(/\/+$/, '');
    const edgeModel = process.env.EXPO_PUBLIC_EDGE_LLM_MODEL?.trim() || 'qwen2-math:1.5b';

    const cloudBase = process.env.EXPO_PUBLIC_LLM_BASE_URL?.trim().replace(/\/+$/, '');
    const cloudKey = process.env.EXPO_PUBLIC_LLM_API_KEY?.trim();
    const cloudModel = process.env.EXPO_PUBLIC_LLM_MODEL?.trim();

    // 优先使用本地/端侧数学强化小模型进行隐私闭环预解，其次使用云端模型
    const useEdge = Boolean(edgeBase);
    const useCloud = !useEdge && Boolean(cloudBase && cloudKey && cloudModel);
    if (!useEdge && !useCloud) return null;

    const inflight = this.inflightPrewarm.get(key);
    if (inflight) return inflight;

    const base = useEdge ? edgeBase! : cloudBase!;
    const model = useEdge ? edgeModel : cloudModel!;
    const endpoint = resolveChatEndpoint(base, useEdge);
    const sourceMode: SolutionCacheEntry['sourceMode'] = useEdge ? 'edge_offline' : 'cloud_prewarm';

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(useCloud && cloudKey ? { Authorization: 'Bearer ' + cloudKey } : {}),
    };

    const customTimeout = Number(process.env.EXPO_PUBLIC_EDGE_LLM_TIMEOUT_MS);
    const timeoutMs =
      Number.isFinite(customTimeout) && customTimeout >= 5000 ? customTimeout : PREWARM_TIMEOUT_MS;

    const task = (async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const compactEdgePrompt = [
          `题干：${problemText.trim()}`,
          '请输出严格合法的 JSON 对象（不要照抄尖括号提示词，不要给出完整解答）：',
          '- steps: 2~4步核心路线，每项包含 stepIndex, goalDescription(用中文一句话说明该步做什么变换，如“利用特征方程展开行列式求特征值”), expectedTokens(数学符号数组)',
          '- commonPitfalls: 1~2条易错陷阱，每项包含 triggerPattern, warningHint(15~40字温和核对提醒)',
          '- sealedFinalAnswer: 字符串，仅放最终数值/表达式以便后端封存',
          '- hintsByStyle: 包含 socratic_question 和 algebra_check 各自的 l1, l2（绝不泄露答案）。',
        ].join('\n');

        const coreStyleList = HINT_STYLES.slice(0, 4)
          .map(s => `${s}(${HINT_STYLE_LABELS[s]})`)
          .join('、');
        const cloudPrompt = [
          `题干：${problemText.trim()}`,
          `请为该题生成结构化解题图谱 JSON（仅返回纯 JSON，确保 LaTeX 反斜杠用双反斜杠或简明文本）：包含 steps 数组(stepIndex, goalDescription, expectedTokens)、commonPitfalls 数组(triggerPattern, warningHint)、sealedFinalAnswer（仅写最终数值/式子）以及 ${coreStyleList} 的 l1/l2/l3 提示。`,
          '规则：steps[].goalDescription 与 l1/l2/l3 严禁出现最终答案或完整结果，只提示步骤方向与易错细节。',
        ].join('\n\n');

        const edgeTokenBudget = /deepseek|r1|reasoner/i.test(model) ? 3072 : 1600;
        const bodyObj: Record<string, unknown> = {
          model,
          stream: false,
          temperature: 0.15,
          max_tokens: useEdge ? edgeTokenBudget : 1200,
          ...(useEdge ? { repeat_penalty: 1.15 } : {}),
          messages: [
            { role: 'system', content: PREWARM_SYSTEM_MESSAGE },
            { role: 'user', content: useEdge ? compactEdgePrompt : cloudPrompt },
          ],
        };

        const response = await fetch(endpoint, {
          method: 'POST',
          signal: controller.signal,
          headers,
          body: JSON.stringify(bodyObj),
        });
        if (!response.ok) return null;

        const completion = (await response.json()) as CompletionResponse;
        const choiceMsg = completion?.choices?.[0]?.message;
        const contentStr = typeof choiceMsg?.content === 'string' ? choiceMsg.content : '';
        const reasoningStr =
          typeof choiceMsg?.reasoning_content === 'string' && choiceMsg.reasoning_content.trim()
            ? choiceMsg.reasoning_content
            : typeof choiceMsg?.reasoning === 'string'
              ? choiceMsg.reasoning
              : '';
        const rawContent = contentStr.trim() ? contentStr : reasoningStr;
        if (!rawContent.trim()) return null;

        // 通道 A：直接解析结构化 JSON 并自动修补 LaTeX 转义 + 补齐全部 6 风格 L0-L3
        const parsed = parseSolutionGraphJson(rawContent);
        if (parsed && (Array.isArray(parsed.steps) || parsed.hintsByStyle)) {
          const rawSfa = parsed.sealedFinalAnswer as unknown;
          const sealedRaw =
            typeof rawSfa === 'string'
              ? rawSfa.trim()
              : rawSfa && typeof rawSfa === 'object'
                ? String(
                    (rawSfa as Record<string, unknown>).answer ??
                      (rawSfa as Record<string, unknown>).tokens ??
                      (rawSfa as Record<string, unknown>).value ??
                      JSON.stringify(rawSfa),
                  ).trim()
                : '';
          const sealed = sealedRaw && !ECHO_PLACEHOLDER_RE.test(sealedRaw) ? sealedRaw : undefined;

          const domainFallback = MATH_DOMAIN_KNOWLEDGE.filter(k =>
            k.pattern.test(problemText),
          ).slice(0, 2);

          const rawSteps = Array.isArray(parsed.steps) ? parsed.steps : [];
          let cleanSteps: SolutionStepSpec[] = rawSteps
            .filter(s => s && cleanModelText(s.goalDescription, 5).length >= 5)
            .map((s, idx) => ({
              stepIndex: typeof s.stepIndex === 'number' ? s.stepIndex : idx,
              goalDescription: stripSpoilerFromGoal(cleanModelText(s.goalDescription, 5), sealed),
              expectedTokens: Array.isArray(s.expectedTokens)
                ? s.expectedTokens.map(String).filter(t => !ECHO_PLACEHOLDER_RE.test(t)).slice(0, 6)
                : extractMathTokensFromText(s.goalDescription),
            }))
            .filter(s => s.goalDescription.length >= 5 && !ECHO_PLACEHOLDER_RE.test(s.goalDescription));

          if (cleanSteps.length === 0) {
            cleanSteps = [
              {
                stepIndex: 0,
                goalDescription: `明确已知条件并围绕「${domainFallback[0]?.clue ?? '核心等式变形'}」起步推导`,
                expectedTokens: extractMathTokensFromText(problemText),
              },
            ];
          }

          const rawPitfalls = Array.isArray(parsed.commonPitfalls) ? parsed.commonPitfalls : [];
          let cleanPitfalls: SolutionPitfallSpec[] = rawPitfalls
            .map(p => ({
              triggerPattern: cleanModelText(p?.triggerPattern, 4),
              warningHint:
                filterSafeHintText(cleanModelText(p?.warningHint, 8), sealed) ??
                domainFallback[0]?.warningHint ??
                '',
            }))
            .filter(p => p.triggerPattern.length > 0 && p.warningHint.length >= 8);
          if (cleanPitfalls.length === 0) {
            cleanPitfalls = domainFallback.map(d => ({
              triggerPattern: d.triggerPattern,
              warningHint: d.warningHint,
            }));
          }

          let normalizedStyles: Partial<Record<HintStyle, StyleHintTiers>> | undefined;
          const rawStylesUnknown = parsed.hintsByStyle as unknown;
          if (Array.isArray(rawStylesUnknown)) {
            normalizedStyles = {};
            for (const item of rawStylesUnknown) {
              if (!item || typeof item !== 'object') continue;
              for (const style of HINT_STYLES) {
                const v = (item as Record<string, unknown>)[style];
                if (v && typeof v === 'object') {
                  normalizedStyles[style] = v as StyleHintTiers;
                } else if (typeof v === 'string') {
                  normalizedStyles[style] = { l1: v };
                }
              }
            }
          } else if (rawStylesUnknown && typeof rawStylesUnknown === 'object') {
            normalizedStyles = rawStylesUnknown as Partial<Record<HintStyle, StyleHintTiers>>;
          }

          const hintsByStyle = buildCompleteHintsFromGraph(
            problemText,
            cleanSteps,
            cleanPitfalls,
            normalizedStyles,
            sealed,
          );

          const entry: SolutionCacheEntry = {
            problemKey: key,
            problemText: problemText.trim(),
            steps: cleanSteps,
            commonPitfalls: cleanPitfalls,
            hintsByStyle,
            ...(sealed ? { sealedFinalAnswer: sealed } : {}),
            sourceMode,
            updatedAt: Date.now(),
          };
          this.entries.set(key, entry);
          return entry;
        }

        // 通道 B：数学强化小模型以自然推导步骤 + \boxed{} 响应时，自动执行 CoT 蒸馏与答案封存
        const distilled = distillSolutionGraphFromMathCoT(problemText, rawContent, sourceMode);
        if (distilled) {
          this.entries.set(key, distilled);
          return distilled;
        }
        return null;
      } catch {
        return null;
      } finally {
        clearTimeout(timeout);
        this.inflightPrewarm.delete(key);
      }
    })();

    this.inflightPrewarm.set(key, task);
    return task;
  }

  public exportAllEntries(): SolutionCacheEntry[] {
    return Array.from(this.entries.values());
  }
}
