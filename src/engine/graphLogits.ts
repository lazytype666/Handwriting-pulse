import { MATH_VOCAB_INDEX } from './mathVocab';

export type GraphGeometricRole =
  | 'logical_connector'
  | 'fraction_bar'
  | 'equation_rel'
  | 'calculus_operator'
  | 'matrix_delimiter'
  | 'probability_rel'
  | 'operand';

export const GRAPH_GEOMETRIC_ROLES: readonly GraphGeometricRole[] = [
  'logical_connector',
  'fraction_bar',
  'equation_rel',
  'calculus_operator',
  'matrix_delimiter',
  'probability_rel',
  'operand',
];

const GRAPH_ROLE_SET = new Set<string>(GRAPH_GEOMETRIC_ROLES);

export interface StepLogitHeadSpec {
  /** 逻辑回归偏置项 β_k ∈ [-6, 4] */
  bias: number;
  /** 期待该步骤出现的归一化进度中心 c_k ∈ [0, 1] */
  progressCenter: number;
  /** 出现即可提升步骤匹配对数似然的符号权重（键为 125 维 MATH_VOCAB 记号） */
  tokenWeights: Record<string, number>;
  /** 几何结构角色加权（如 fraction_bar / calculus_operator / matrix_delimiter） */
  roleWeights: Partial<Record<GraphGeometricRole, number>>;
}

export interface PitfallLogitHeadSpec {
  /** 逻辑回归偏置项 β ∈ [-6, 4] */
  bias: number;
  /** 触发易错警告的概率阈值 τ ∈ [0.45, 0.9] */
  threshold: number;
  /** 触发判定的最少行数下限（未写到该行之前抑制误报） */
  minLineCount: number;
  /** 行数推进权重 w_line ∈ [-2, 2] */
  lineCountWeight: number;
  /** 观测到对应符号提高易错分数的正权重（P_obs(v) 加权） */
  presentWeights: Record<string, number>;
  /** 缺失对应关键符号提高易错分数的权重（1 - P_obs(v) 加权） */
  missingWeights: Record<string, number>;
  /** 结构角色权重（正值增强触发，负值表示该结构已书写则缓解警报） */
  roleWeights: Partial<Record<GraphGeometricRole, number>>;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function tokenizeSnippetToVocab(text: string): string[] {
  const tokens: string[] = [];
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === '\\') {
      let j = i + 1;
      while (j < text.length && /[a-zA-Z]/.test(text[j])) j += 1;
      const cmd = text.slice(i, j);
      if (MATH_VOCAB_INDEX.has(cmd) && cmd !== '<blank>') {
        tokens.push(cmd);
      }
      i = Math.max(j, i + 1);
      continue;
    }
    if (MATH_VOCAB_INDEX.has(ch) && ch !== '<blank>') {
      tokens.push(ch);
    }
    i += 1;
  }
  return tokens;
}

const SKIP_NOISE_CHARS = new Set(['{', '}', '(', ')', ' ', '\t', '\n']);

/**
 * 将模型输出或解析得到的 token -> weight 映射规整至 125 维 `MATH_VOCAB` 合法记号空间。
 */
export function sanitizeTokenWeightMap(raw: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out;
  for (const [kRaw, vRaw] of Object.entries(raw as Record<string, unknown>)) {
    const w = Number(vRaw);
    if (!Number.isFinite(w) || Math.abs(w) < 1e-4) continue;
    const key = kRaw.trim();
    if (!key || key === '<blank>') continue;
    const clampedW = Number(clamp(w, -4, 4).toFixed(3));
    if (MATH_VOCAB_INDEX.has(key)) {
      out[key] = clampedW;
      continue;
    }
    const sub = tokenizeSnippetToVocab(key).filter(t => !SKIP_NOISE_CHARS.has(t));
    if (sub.length > 0) {
      const scaledW = Number((clampedW / Math.min(sub.length, 3)).toFixed(3));
      for (const tok of sub.slice(0, 4)) {
        out[tok] = Number(clamp((out[tok] ?? 0) + scaledW, -4, 4).toFixed(3));
      }
    }
  }
  return out;
}

export function sanitizeRoleWeightMap(
  raw: unknown,
): Partial<Record<GraphGeometricRole, number>> {
  const out: Partial<Record<GraphGeometricRole, number>> = {};
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out;
  for (const [kRaw, vRaw] of Object.entries(raw as Record<string, unknown>)) {
    const k = kRaw.trim();
    if (!GRAPH_ROLE_SET.has(k)) continue;
    const w = Number(vRaw);
    if (!Number.isFinite(w) || Math.abs(w) < 1e-4) continue;
    out[k as GraphGeometricRole] = Number(clamp(w, -4, 4).toFixed(3));
  }
  return out;
}

/**
 * 从步骤描述与期望符号自动编译连续 `StepLogitHeadSpec`（当大模型未显式生成或样本预加载时兜底）。
 */
export function compileDefaultStepLogits(
  expectedTokens: string[],
  stepIndex: number,
  totalSteps: number,
  goalDescription = '',
): StepLogitHeadSpec {
  const tokenWeights: Record<string, number> = {};
  const roleWeights: Partial<Record<GraphGeometricRole, number>> = {};
  const combined = `${expectedTokens.join(' ')} ${goalDescription}`;

  for (const rawTok of expectedTokens) {
    const trimmed = rawTok.trim();
    if (MATH_VOCAB_INDEX.has(trimmed) && trimmed !== '<blank>') {
      tokenWeights[trimmed] = trimmed === '=' ? 0.9 : 1.25;
      continue;
    }
    for (const sub of tokenizeSnippetToVocab(trimmed)) {
      if (!SKIP_NOISE_CHARS.has(sub)) {
        tokenWeights[sub] = Math.max(tokenWeights[sub] ?? 0, sub.startsWith('\\') ? 1.15 : 0.7);
      }
    }
  }

  if (/(?:分式|分数|比值|分子|分母|\\frac|\/)/.test(combined)) {
    tokenWeights['\\frac'] = Math.max(tokenWeights['\\frac'] ?? 0, 1.0);
    roleWeights.fraction_bar = 1.2;
  }
  if (/(?:积分|原函数|上下限|导数|极限|\\int|\\oint|\\lim|\\partial)/.test(combined)) {
    roleWeights.calculus_operator = 1.25;
  }
  if (/(?:矩阵|行列式|特征|秩|正交|\\matrix|\\det|\|A\|)/.test(combined)) {
    roleWeights.matrix_delimiter = 1.3;
  }
  if (/(?:概率|贝叶斯|条件|正态|\\Phi|\\cap|\\cup)/.test(combined)) {
    roleWeights.probability_rel = 1.15;
  }
  if (/(?:=|等式|通解|特解|特征方程)/.test(combined)) {
    tokenWeights['='] = Math.max(tokenWeights['='] ?? 0, 0.85);
    roleWeights.equation_rel = 0.9;
  }
  if (Object.keys(tokenWeights).length === 0) {
    tokenWeights['='] = 0.8;
  }

  const denom = Math.max(1, totalSteps - 1);
  return {
    bias: -0.95,
    progressCenter: Number(clamp(totalSteps > 1 ? stepIndex / denom : 0.5, 0, 1).toFixed(3)),
    tokenWeights,
    roleWeights,
  };
}

/**
 * 规整大模型生成的 stepLogits，若缺失或为空则自动用规则编译完整向量。
 */
export function sanitizeStepLogits(
  raw: unknown,
  expectedTokens: string[],
  stepIndex: number,
  totalSteps: number,
  goalDescription = '',
): StepLogitHeadSpec {
  const fallback = compileDefaultStepLogits(expectedTokens, stepIndex, totalSteps, goalDescription);
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return fallback;
  }
  const obj = raw as Record<string, unknown>;
  const tokenWeights = sanitizeTokenWeightMap(obj.tokenWeights);
  const roleWeights = sanitizeRoleWeightMap(obj.roleWeights);
  if (Object.keys(tokenWeights).length === 0 && Object.keys(roleWeights).length === 0) {
    return fallback;
  }
  const rawBias = Number(obj.bias);
  const rawCenter = Number(obj.progressCenter);
  return {
    bias: Number((Number.isFinite(rawBias) ? clamp(rawBias, -6, 4) : fallback.bias).toFixed(3)),
    progressCenter: Number(
      (Number.isFinite(rawCenter) ? clamp(rawCenter, 0, 1) : fallback.progressCenter).toFixed(3),
    ),
    tokenWeights: Object.keys(tokenWeights).length > 0 ? tokenWeights : fallback.tokenWeights,
    roleWeights: Object.keys(roleWeights).length > 0 ? roleWeights : fallback.roleWeights,
  };
}

/**
 * 从自然语言易错模式与提示自动编译连续 `PitfallLogitHeadSpec`。
 * 兼容含参矩阵临界零点、积分因子/极坐标雅可比/变限代换、泰勒阶数、全概率分母求和与通用代换/正负号漏项等场景。
 */
export function compileDefaultPitfallLogits(
  triggerPattern: string,
  warningHint = '',
): PitfallLogitHeadSpec {
  const text = `${triggerPattern} ${warningHint}`.toLowerCase();
  const presentWeights: Record<string, number> = {};
  const missingWeights: Record<string, number> = {};
  const roleWeights: Partial<Record<GraphGeometricRole, number>> = {};

  let bias = -2.25;
  let threshold = 0.62;
  let minLineCount = 2;
  let lineCountWeight = 0.42;

  if (/\|a\|=0|行列式|临界|含参/.test(text)) {
    presentWeights['='] = 0.65;
    missingWeights['0'] = 1.45;
    roleWeights.matrix_delimiter = -1.35;
    bias = -1.45;
    minLineCount = 2;
    lineCountWeight = 0.55;
  } else if (/阶数不够|截断|洛必达|泰勒|高阶/.test(text)) {
    presentWeights['\\frac'] = 1.15;
    presentWeights['='] = 0.45;
    missingWeights['^'] = 0.85;
    roleWeights.fraction_bar = 0.9;
    minLineCount = 3;
    lineCountWeight = 0.48;
    bias = -2.35;
  } else if (/全概率|分母|先验权重|标准差|贝叶斯|归一/.test(text)) {
    presentWeights['='] = 0.7;
    missingWeights['\\frac'] = 1.35;
    missingWeights['\\sum'] = 1.1;
    roleWeights.fraction_bar = -1.3;
    bias = -1.6;
    minLineCount = 2;
  } else if (/雅可比|polar|极坐标|因子\s*r/.test(text)) {
    presentWeights['\\int'] = 0.85;
    missingWeights['r'] = 1.55;
    bias = -1.75;
    minLineCount = 2;
  } else if (/特征根|重数|x\^k|积分因子|特解|常数\s*c/.test(text)) {
    presentWeights['='] = 0.7;
    missingWeights['e'] = 0.9;
    missingWeights['C'] = 0.85;
    bias = -1.9;
    minLineCount = 2;
  } else if (/正交|施密特|单位化/.test(text)) {
    presentWeights['='] = 0.6;
    missingWeights['\\frac'] = 1.2;
    missingWeights['\\sqrt'] = 0.95;
    roleWeights.fraction_bar = -1.1;
    bias = -1.75;
    minLineCount = 2;
  } else if (/除以2|分式|分母|系数/.test(text)) {
    presentWeights['='] = 0.75;
    missingWeights['\\frac'] = 1.4;
    roleWeights.fraction_bar = -1.25;
    bias = -1.65;
    minLineCount = 2;
  } else {
    presentWeights['='] = 0.65;
    for (const tok of tokenizeSnippetToVocab(triggerPattern)) {
      if (!SKIP_NOISE_CHARS.has(tok) && tok !== '=') {
        missingWeights[tok] = 0.9;
      }
    }
    bias = -2.1;
    threshold = 0.66;
  }

  return {
    bias: Number(clamp(bias, -6, 4).toFixed(3)),
    threshold: Number(clamp(threshold, 0.45, 0.9).toFixed(3)),
    minLineCount,
    lineCountWeight: Number(clamp(lineCountWeight, -2, 2).toFixed(3)),
    presentWeights,
    missingWeights,
    roleWeights,
  };
}

/**
 * 规整大模型返回的 `pitfallLogits`；若某道题未生成有效权重，则自动编译语义兜底权重向量。
 */
export function sanitizePitfallLogits(
  raw: unknown,
  triggerPattern: string,
  warningHint = '',
): PitfallLogitHeadSpec {
  const fallback = compileDefaultPitfallLogits(triggerPattern, warningHint);
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return fallback;
  }
  const obj = raw as Record<string, unknown>;
  const presentWeights = sanitizeTokenWeightMap(obj.presentWeights);
  const missingWeights = sanitizeTokenWeightMap(obj.missingWeights);
  const roleWeights = sanitizeRoleWeightMap(obj.roleWeights);
  if (
    Object.keys(presentWeights).length === 0 &&
    Object.keys(missingWeights).length === 0 &&
    Object.keys(roleWeights).length === 0
  ) {
    return fallback;
  }
  const rawBias = Number(obj.bias);
  const rawTh = Number(obj.threshold);
  const rawMinLine = Number(obj.minLineCount);
  const rawLineW = Number(obj.lineCountWeight);
  return {
    bias: Number((Number.isFinite(rawBias) ? clamp(rawBias, -6, 4) : fallback.bias).toFixed(3)),
    threshold: Number(
      (Number.isFinite(rawTh) ? clamp(rawTh, 0.45, 0.9) : fallback.threshold).toFixed(3),
    ),
    minLineCount: Number.isFinite(rawMinLine) ? Math.max(1, Math.min(5, Math.round(rawMinLine))) : fallback.minLineCount,
    lineCountWeight: Number(
      (Number.isFinite(rawLineW) ? clamp(rawLineW, -2, 2) : fallback.lineCountWeight).toFixed(3),
    ),
    presentWeights,
    missingWeights,
    roleWeights,
  };
}
