import type { SolutionCacheEntry } from './solutionCache';
import { normalizeProblemKey } from './solutionCache';

export interface SampleProblemSpec {
  id: string;
  label: string;
  category: string;
  problemText: string;
  cacheEntry: SolutionCacheEntry;
}

function buildEntry(
  problemText: string,
  steps: SolutionCacheEntry['steps'],
  commonPitfalls: SolutionCacheEntry['commonPitfalls'],
  hintsByStyle: SolutionCacheEntry['hintsByStyle'],
  sealedFinalAnswer: string,
): SolutionCacheEntry {
  return {
    problemKey: normalizeProblemKey(problemText),
    problemText,
    steps,
    commonPitfalls,
    hintsByStyle,
    sealedFinalAnswer,
    sourceMode: 'edge_offline',
    updatedAt: Date.now(),
  };
}

/**
 * 大学理工科数学演示题库与离线解题图谱（全面覆盖高等数学、线性代数与概率论与数理统计）。
 * 全部题目为公开典型大学教学例题，严守匿名与防剧透规范。
 */
export const SAMPLE_PROBLEMS: SampleProblemSpec[] = [
  {
    id: 'calc_taylor_limit',
    label: '泰勒展开与未定式极限',
    category: '高等数学',
    problemText: '求极限 lim(x→0) [e^x - 1 - x - (1/2)x^2] / (x - sin x)。',
    cacheEntry: buildEntry(
      '求极限 lim(x→0) [e^x - 1 - x - (1/2)x^2] / (x - sin x)。',
      [
        { stepIndex: 0, goalDescription: '先判定为 0/0 型未定式，确定分母 (x - sin x) 的等价无穷小阶数', expectedTokens: ['\\to', '\\frac', 'x'] },
        { stepIndex: 1, goalDescription: '依据“上下同阶”原则，将分子 e^x 与分母 sin x 展开至三次项 x³', expectedTokens: ['=', '+', '\\frac'] },
        { stepIndex: 2, goalDescription: '抵消低阶项后提取最高次主部系数之比完成化简', expectedTokens: ['=', '\\frac'] },
      ],
      [
        {
          triggerPattern: '截断阶数不够，或反复使用洛必达法则导致计算繁琐出错',
          warningHint: '观察分母 x - sin x 的最低非零阶数是多少次幂？如果只把 e^x 展开到平方项，会不会刚好把主部抵消光？',
        },
      ],
      {
        socratic_question: {
          l1: '遇到 e^x 和 sin x 同时出现的 0/0 型极限，硬算三阶洛必达还是按阶数泰勒展开更稳妥？',
          l2: '先看分母 x - sin x 展开后最低留下的 x 次幂是多少，再决定分子 e^x 至少得展开到第几阶。',
          l3: '按照“上下同阶”确定三次主部阶数后，将分子分母同时按泰勒公式替换至 x³ 阶并加高阶无穷小余项即可比较系数。',
        },
        condition_reminder: {
          l1: '注意别忘了减数里有 -1 - x - (1/2)x^2，前三项在分子展开中会被恰好抵消！',
          l2: '乘除因式可以直接等价无穷小代换，但分母 x - sin x 是加减运算，切忌单独把 sin x 替换成 x！',
          l3: '利用加减项不可随便局部等价的准则，对指数和正弦同步采用同阶麦克劳林展开。',
        },
        algebra_check: {
          l1: '核对一下 sin x 展开到 x³ 项前面的正负号，以及 x - sin x 去括号后三次项是否变成正号。',
          l2: '注意 e^x 三次项系数是 1 除以几的阶乘？别把 3! 误写成 3 啦。',
          l3: '分子抵消掉前三项后只留 x³ 项，检查分母三次项的符号与阶乘倒数是否都已翻成正比。',
        },
        step_decompose: {
          l1: '第一步先只算分母：单独写出 sin x 到三阶的展开式，看看 x - sin x 等价于几倍 x³。',
          l2: '知道目标阶数是三次之后，第二步专心写出 e^x 到 x³ 阶的多项式再减去分子其余项。',
          l3: '拆成三步走：①确定分母等价阶数；②分子做同阶展开抵消前几项；③化简同阶系数比例。',
        },
      },
      '1',
    ),
  },
  {
    id: 'calc_var_integral_extrema',
    label: '变上限积分与极值点',
    category: '高等数学',
    problemText: '设函数 F(x) = ∫(0→x) (t² - 1)e^(-t) dt，求 F(x) 的单调区间与极值点。',
    cacheEntry: buildEntry(
      '设函数 F(x) = ∫(0→x) (t² - 1)e^(-t) dt，求 F(x) 的单调区间与极值点。',
      [
        { stepIndex: 0, goalDescription: '应用变上限积分求导公式直接写出 F′(x)', expectedTokens: ['\\int', '=', 'x'] },
        { stepIndex: 1, goalDescription: '令导函数 F′(x) = 0 求解全部驻点', expectedTokens: ['=', '0', '1'] },
        { stepIndex: 2, goalDescription: '利用指数因子恒正检验驻点左右导数正负符号变化', expectedTokens: ['=', '1'] },
      ],
      [
        {
          triggerPattern: '先试图用分部积分把 F(x) 显式积出来，或求导后漏掉替换积分变量',
          warningHint: '题目问的是“单调区间与极值点”，真的需要费力先算出原积分函数吗？直接对 F(x) 求导会发生什么？',
        },
      ],
      {
        socratic_question: {
          l1: '看到求变限积分函数的单调性和极值，第一步应该先对 F(x) 积分还是先求导？',
          l2: '利用变限积分求导定理得出 F′(x) 后，其中 e^(-x) 项在实数轴上的正负号对单调性判断有影响吗？',
          l3: '直接用上限变量替换被积表达式求出一阶导，消去恒正的指数因子后，只需考察二次多项式因子的穿根正负变号即可。',
        },
        condition_reminder: {
          l1: '注意区分积分变量 t 与上限自变量 x：求导后表达式里不应该再出现 t。',
          l2: '判断极值不仅要解 F′(x) = 0 的两个根，还要核对零点两侧导数的正负翻转方向。',
          l3: '抓住指数函数恒大于零的性质简化不等号，再沿着数轴按正负区间对应出单调性与极值。',
        },
        algebra_check: {
          l1: '解 x² - 1 = 0 时别丢掉负根！两个驻点在数轴上都要划出区间边界。',
          l2: '代特值检验 -1 左侧与 (-1, 1) 之间导数值正负时，注意 x²-1 是开口向上的二次式。',
          l3: '核对一下极值点究竟对应极大还是极小，记得区分“先增后减”与“先减后增”。',
        },
        step_decompose: {
          l1: '先只干第一件：套用变上限求导公式，一行写出 F′(x) 的表达式。',
          l2: '第二步忽略恒正因子 e^(-x)，只解一个二次代数方程找驻点并画一条简单数轴标注正负。',
          l3: '分三步落地：①变限求导得 F′(x)；②令导数为零找两驻点；③按数轴符号划分递增递减与极值类型。',
        },
      },
      'x=-1 极大值点，x=1 极小值点',
    ),
  },
  {
    id: 'calc_ode_first_order',
    label: '一阶线性微分方程特解',
    category: '高等数学',
    problemText: '求微分方程 xy′ + y = x ln x 满足初始条件 y(1) = 0 的特解。',
    cacheEntry: buildEntry(
      '求微分方程 xy′ + y = x ln x 满足初始条件 y(1) = 0 的特解。',
      [
        { stepIndex: 0, goalDescription: '观察方程左侧即乘积导数 (xy)′，或两边同除 x 化为标准一阶线性形', expectedTokens: ['=', '+', 'x'] },
        { stepIndex: 1, goalDescription: '两边积分并用分部积分公式计算 ∫ x ln x dx', expectedTokens: ['\\int', '=', '\\frac'] },
        { stepIndex: 2, goalDescription: '利用初始条件 y(1) = 0 定出任意常数 C 并显式写出 y(x)', expectedTokens: ['=', '0', '1'] },
      ],
      [
        {
          triggerPattern: '直接套通解公式前忘记先除以 x，或对 x ln x 误用求导代替分部积分',
          warningHint: '看一眼等号左边的 xy′ + y，它正好是哪两个函数乘积的导数？如果要用通解公式，y′ 前面的系数必须化成什么？',
        },
      ],
      {
        socratic_question: {
          l1: '观察方程左边 xy′ + y，能不能逆用乘积求导法则把它合并成某一个整体的导数？',
          l2: '合并出 (xy)′ = x ln x 后两边同时积分，右边这个幂函数乘对数的积分该用哪种积分法？',
          l3: '将左边凑成乘积微分 (xy)′ 或先同除 x 套线性公式，右侧利用“反对幂指三”选分部积分求通解，再代初值定常数。',
        },
        condition_reminder: {
          l1: '题干给出了初值条件 y(1) = 0 且存在 ln x（隐含定义域 x > 0），别忘了利用 x = 1 定出唯一常数 C。',
          l2: '如果选用一阶线性公式 y′ + P(x)y = Q(x)，请先确保已将方程首项系数 x 除到右边。',
          l3: '记住积分常数 C 必须在求通解之后再代入初值确定，不能等到除以 x 之后才随手在末尾加 +C。',
        },
        algebra_check: {
          l1: '用分部积分算 ∫ x ln x dx 时，选定 u = ln x、dv = x dx，核对一下 v = (1/2)x² 的系数 1/2 有没有带进后面减去的新积分里。',
          l2: '第二步减号里的积分 ∫ (1/2)x² · (1/x) dx 化简并再积分后，分母应该变成多少？',
          l3: '算出特解后，可以把 y(1) 和原方程左端 xy′ + y 代回去快速逆推检验是否正好还原成 x ln x。',
        },
        step_decompose: {
          l1: '先不忙套长公式，试着写下一行：因为 (xy)′ = xy′ + y，所以 (xy)′ 等于什么？',
          l2: '接下来单独在一旁只算一个不定积分：∫ x ln x dx，算出结果后再加上常数 C 赋给 xy。',
          l3: '分三步完成：①左侧凑微分合并；②分部积分求右侧并带 +C；③代 y(1)=0 求 C 并除 x。',
        },
      },
      'y = (1/2)x ln x - x/4 + 1/(4x)',
    ),
  },
  {
    id: 'la_parametric_linear_system',
    label: '含参线性方程组解的讨论',
    category: '线性代数',
    problemText: '设系数矩阵 A = [[1,1,λ],[1,λ,1],[λ,1,1]]，右端向量 b = (1, λ, λ²)ᵀ，讨论 λ 取何值时方程组 Ax = b 无解、有唯一解、有无穷多解？',
    cacheEntry: buildEntry(
      '设系数矩阵 A = [[1,1,λ],[1,λ,1],[λ,1,1]]，右端向量 b = (1, λ, λ²)ᵀ，讨论 λ 取何值时方程组 Ax = b 无解、有唯一解、有无穷多解？',
      [
        { stepIndex: 0, goalDescription: 'A 为三阶方阵，先计算系数行列式 |A| 并因式分解找出临界点', expectedTokens: ['|', '=', 'det'] },
        { stepIndex: 1, goalDescription: '判定当 |A| ≠ 0 时矩阵可逆，原非齐次方程组必有唯一解', expectedTokens: ['=', '0'] },
        { stepIndex: 2, goalDescription: '分别将各临界 λ 值回代增广矩阵 [A|b] 做初等行变换比较 r(A) 与 r(A,b)', expectedTokens: ['\\matrix', '='] },
      ],
      [
        {
          triggerPattern: '未求系数行列式 |A|=0 漏讨论零点，或直接带参数行变换误除以含 (λ-1) 的因式',
          warningHint: '未知数个数和方程个数都是 3（A 是方阵），先算一下系数行列式 |A| 有多省事？注意不要直接带参数同除以可能是 0 的表达式！',
        },
      ],
      {
        socratic_question: {
          l1: '因为系数矩阵 A 是 3×3 方阵，在分类讨论 Ax = b 的解之前，求哪个数值能瞬间锁定唯一解和非唯一解的分界线？',
          l2: '利用行各元素之和相等把 |A| 分解成 (λ + 2)(λ - 1)² 后，对于使行列式为零的那两个 λ 值该如何分别判断无解还是无穷多解？',
          l3: '遵循“方阵先算 |A| 定分界 → 非零即唯一解 → 零点逐个回代增广矩阵比对 r(A) 与 r(A,b)”的标准线性代数流程推进。',
        },
        condition_reminder: {
          l1: '注意题干给的方程组既要判断唯一解、又要区分无解和无穷多解：唯一解条件看 r(A)=r(A,b)=3（即 |A|≠0），无解与无穷多解要看增广矩阵的秩是否跳变。',
          l2: '利用方阵每行或每列累加相等的技巧计算 |A|，比硬按对角线法则展开更容易看出公因式 (λ + 2)！',
          l3: '回代特殊参数值 λ = 1 与 λ = -2 后，增广矩阵已经变成纯数字矩阵，初等行变换只需做整数消元。',
        },
        algebra_check: {
          l1: '因式分解三阶行列式后检查有没有漏掉常数或符号：全部加到第一列提取 λ + 2 后，剩余二阶块化成上三角对角线元素是什么？',
          l2: '特别关注右端向量 b = (1, λ, λ²)ᵀ：代入 λ = 1 和 λ = -2 时向量最后一项 λ² 的符号千万算稳！',
          l3: '若得出无解，核对增广矩阵最后一行是不是呈现 [0 0 0 | 非零数] 的矛盾形式。',
        },
        step_decompose: {
          l1: '第一步先暂时不管右端 b，专心写出系数行列式 |A| 并因式分解出全部临界 λ 值。',
          l2: '写出“当 λ 不等于这些临界值时，|A| ≠ 0，有唯一解”；再分别划出两个数字支线单独处理。',
          l3: '拆成三步走：①算 |A| 分界；②给出唯一解区间；③代入临界值做数字增广矩阵行阶梯化判秩。',
        },
      },
      'λ≠1 且 λ≠-2 唯一解；λ=1 无穷多解；λ=-2 无解',
    ),
  },
  {
    id: 'la_symmetric_orthogonal_diag',
    label: '实对称矩阵正交对角化',
    category: '线性代数',
    problemText: '已知三阶实对称矩阵 A 的特征值为 λ₁ = 6, λ₂ = λ₃ = 3，对应于 λ₁ = 6 的特征向量为 α₁ = (1, 1, 1)ᵀ，试理清求正交矩阵 Q 使 QᵀAQ = Λ 及反求矩阵 A 的完整步骤。',
    cacheEntry: buildEntry(
      '已知三阶实对称矩阵 A 的特征值为 λ₁ = 6, λ₂ = λ₃ = 3，对应于 λ₁ = 6 的特征向量为 α₁ = (1, 1, 1)ᵀ，试理清求正交矩阵 Q 使 QᵀAQ = Λ 及反求矩阵 A 的完整步骤。',
      [
        { stepIndex: 0, goalDescription: '利用实对称矩阵“不同特征值对应特征向量正交”建立方程 α₁ᵀx = 0 求解 λ=3 的基础解系', expectedTokens: ['=', '0', '\\matrix'] },
        { stepIndex: 1, goalDescription: '对重特征值 λ=3 的两个线性无关向量执行施密特正交化并全体单位化拼成正交矩阵 Q', expectedTokens: ['=', '\\frac', '\\sqrt'] },
        { stepIndex: 2, goalDescription: '由 QᵀAQ = Λ 与 Q⁻¹ = Qᵀ 推导矩阵还原公式 A = QΛQᵀ', expectedTokens: ['=', 'Q'] },
      ],
      [
        {
          triggerPattern: '多重特征向量忘记先正交化直接单位化，或漏用不同特征值正交性',
          warningHint: '题干没直接给矩阵 A 的元素，只给了 λ₁=6 的特征向量 α₁。想想“实对称矩阵”对于不同特征值的特征向量有一条什么黄金性质？',
        },
      ],
      {
        socratic_question: {
          l1: '手里既没有 A 的具体数字元素，却给了“实对称矩阵”和 λ₁ ≠ λ₂ 的条件，怎么倒推出 λ=3 对应的特征向量？',
          l2: '通过正交方程 x₁ + x₂ + x₃ = 0 找到二重根的两个无关向量后，它们之间天然互相垂直吗？组装 Q 前还缺什么操作？',
          l3: '利用不同特征值正交性求特征子空间基础解系，重根内施密特正交后逐列单位化得正交阵 Q，最终由正交对角化逆推还原 A。',
        },
        condition_reminder: {
          l1: '别漏掉核心条件“实对称矩阵”：第一，不同特征值的特征向量正交；第二，二重特征值必有两个线性无关的特征向量。',
          l2: '正交矩阵 Q 要求各列既两两正交又要长度为 1（单位化），且特征值在 Λ 对角线上的顺序必须和 Q 中向量列序一一对应。',
          l3: '由 QᵀAQ = Λ 反求 A 时，注意正交阵满足 Q⁻¹ = Qᵀ，左乘 Q、右乘 Qᵀ 即可化开，切勿把转置左右乘反。',
        },
        algebra_check: {
          l1: '检查求重根子空间解向量时，把自由变量分别取 (1,0) 和 (0,1) 得到的两个基础向量内积为零吗？若非零就要先正交化！',
          l2: '做施密特正交化时，投影系数分子是内积 (β₁, α₂)、分母是 (β₁, β₁)，写分母时容易误算成范数平方根而忘记平方。',
          l3: '单位化各列后验算一下：每列平方和都等于 1，且各列对应位置内积都严格等于 0。',
        },
        step_decompose: {
          l1: '第一步先只写出正交条件：α₁ᵀx = 0 即 x₁+x₂+x₃=0，任意找出它的两个基础解向量。',
          l2: '第二步在组合成矩阵 Q 之前，只盯着 λ=3 那两个解向量做一次施密特正交化，再把全部三列单位化。',
          l3: '拆三步落地：①由正交方程求二重特征向量；②施密特正交并单位化建 Q；③写 A = QΛQᵀ。',
        },
      },
      'Q 的各列为正交单位化后的特征向量组，A = Q diag(6,3,3) Q^T',
    ),
  },
  {
    id: 'prob_bayes_posterior',
    label: '全概率与贝叶斯后验推断',
    category: '概率统计',
    problemText: '甲、乙、丙三条流水线的产量占比分别为 0.5, 0.3, 0.2，次品率分别为 0.02, 0.03, 0.05。现随机抽取一件产品发现是次品 B，求该次品出自丙线的条件概率 P(A₃ | B)。',
    cacheEntry: buildEntry(
      '甲、乙、丙三条流水线的产量占比分别为 0.5, 0.3, 0.2，次品率分别为 0.02, 0.03, 0.05。现随机抽取一件产品发现是次品 B，求该次品出自丙线的条件概率 P(A₃ | B)。',
      [
        { stepIndex: 0, goalDescription: '定义完备事件组 A₁, A₂, A₃（产地）与目标事件 B（次品），分清先验 P(Aᵢ) 与条件 P(B|Aᵢ)', expectedTokens: ['P(', '=', '0'] },
        { stepIndex: 1, goalDescription: '套用全概率公式 P(B) = ∑ P(Aᵢ)P(B|Aᵢ) 计算抽取次品的总概率', expectedTokens: ['\\sum', '=', '+', '0'] },
        { stepIndex: 2, goalDescription: '代入贝叶斯公式计算后验条件概率 P(A₃ | B) = P(A₃)P(B|A₃) / P(B)', expectedTokens: ['=', '\\frac'] },
      ],
      [
        {
          triggerPattern: '全概率分母漏乘各生产线产量占比先验权重，或将 P(B|A₃) 误当 P(A₃|B)',
          warningHint: '看清所求概率的竖杠前后顺序：已知“抽到了次品”，求“它来自丙线”是先验还是后验逆推？分母需要把三家的次品率直接相加还是要按各自产量占比加权求和？',
        },
      ],
      {
        socratic_question: {
          l1: '题干中给出的 0.05 是“已知来自丙线则是次品”的概率，我们要算的是已知“结果是次品”，反推“原因来自丙线”的概率，应该用哪两个经典公式配合？',
          l2: '计算样本全体抽到次品的总概率 P(B) 时，甲、乙、丙三条线的产量规模不同，它们的贡献权重相等吗？',
          l3: '先以完备事件组划分三条产地线，利用全概率公式按产量占比加权求出次品总概率作为分母，再通过贝叶斯公式比出丙线的后验份额。',
        },
        condition_reminder: {
          l1: '三条线产量占比相加正好 0.5 + 0.3 + 0.2 = 1，正好构成概率空间的一组完备事件划分。',
          l2: '题干分别给了 3 个无条件先验概率 P(Aᵢ) 和 3 个正向条件概率 P(B|Aᵢ)，注意竖杠两边含义绝不可互换。',
          l3: '把分子丙线贡献 P(A₃)P(B|A₃) 与全概率公式求得的分母总和组合起来写成条件概率分式。',
        },
        algebra_check: {
          l1: '计算 P(A₁)P(B|A₁) 等乘积项时小数点位数很多（0.5×0.02 等），对齐小数位数核对有没有少一个零！',
          l2: '求分母 P(B) 时共有几项加权积？三条线都别漏加，尤其丙线自己也是全概率分母里的一部分。',
          l3: '算出来的后验概率应该合理地介于 0 到 1 之间；因为丙线次品率明显高于甲乙，其后验占比一般会略高于它原先的产量份额 0.2，可以用这个直觉快速验算！',
        },
        step_decompose: {
          l1: '第一步先在草稿纸上列三对已知量：清楚写下 P(A₁)、P(A₂)、P(A₃) 和各自对应的 P(B|Aᵢ)。',
          l2: '第二步只求分母 P(B)：按产量 × 次品率把三条线分别乘出来再加总。',
          l3: '分三步走：①标明先验与正向条件概率；②全概率算总次品概率分母；③丙线单项除以总概率得后验比例。',
        },
      },
      '10 / 29',
    ),
  },
];
