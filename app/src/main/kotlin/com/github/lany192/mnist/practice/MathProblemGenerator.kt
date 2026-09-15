package com.github.lany192.mnist.practice

import com.github.lany192.mnist.recognize.MainActivity
import kotlin.random.Random

/**
 * 小学一~六年级口算题的生成器。
 *
 * 两条贯穿全文的约束：
 *
 * 1. **当前答案只能是非负整数且 ≤ [MAX_ANSWER]**。识别链路现在已经支持 0-9 和小数点，
 *    但 [Problem.answer] 仍是 `Int`，改成分数/小数会同时牵动跨页 Intent、Room schema 和迁移；
 *    在本轮改造完成前，五六年级题面仍允许出现小数、分数、百分数，但**答案一律构造为整数**。
 *    位数上限来自"笔画宽/数字高"比例对识别精度的影响（见 MainActivity.MAX_DIGITS 的注释）。
 *
 * 2. **先定答案、再反推操作数**，不做"随机生成后筛"。除法、小数、分数的可行组合非常稀疏：
 *    小数乘法要求末位与乘数之积 ≡ 0 (mod 10)，枚举后只剩两个解族；筛法要么命中率极低，
 *    要么根本生成不出。因此每个题型都给出闭式构造。
 *
 * 本类不依赖任何 Android API，出题逻辑可在 JVM 单元测试中直接验证。
 * **不要在这个文件里 import 任何 `android.*` 的东西** —— 与 InkSegmenter 同一条硬边界，
 * 一旦引入，MathProblemGeneratorTest 就跑不起来了。
 */
object MathProblemGenerator {
    const val DEFAULT_COUNT = 10

    /**
     * 答案位数上限。与 `MainActivity.MAX_DIGITS` 是同一个物理约束，那边是 private，
     * 这里显式再声明一次，供练习页做 `recognize(bitmap, ANSWER_DIGITS)` 的入参。
     */
    const val ANSWER_DIGITS = 4
    private const val MAX_ANSWER = 9999

    /**
     * 去重重试上限。低年级题面空间最小（一年级加法只有约 190 种），
     * 无上限的 `while (!seen.add(...))` 在 count 被调大时会死循环。
     */
    private const val DUPLICATE_RETRY = 20

    /** 分数题的分母上限。再大就超出手算练习的合理范围了。 */
    private const val MAX_DENOMINATOR = 12

    /**
     * 生成 [count] 道题。
     *
     * 题型按 [makersFor] 的顺序**轮转**取用，而不是每题独立随机抽题型：二年级只有 4 种题型，
     * 独立抽取时"10 道全是加法"这类分布抖动很常见，教学上不可接受。轮转保证每种题型出现的
     * 次数相差不超过 1。
     *
     * @param random 可注入，便于单元测试用固定种子复现。**实现内部所有取随机数的地方都必须
     *   使用这个实例**，用 `Random.Default` 会让"固定种子可复现"的单测随机失败。
     */
    fun generateSet(
        grade: Grade,
        count: Int = DEFAULT_COUNT,
        random: Random = Random.Default
    ): List<Problem> {
        require(count > 0) { "题数必须为正：$count" }
        val makers = makersFor(grade)
        val seen = HashSet<String>(count * 2)
        val problems = ArrayList<Problem>(count)
        var cursor = 0
        while (problems.size < count) {
            val maker = makers[cursor % makers.size]
            cursor++
            var candidate = maker(random)
            var attempt = 0
            // seen.add 在碰到重复项时返回 false 且**不插入**，所以循环正常退出时必然已插入成功；
            // 唯一的例外是重试耗尽 —— 那时宁可保留一道重复题，也要保证函数一定终止。
            while (!seen.add(candidate.expression) && attempt < DUPLICATE_RETRY) {
                candidate = maker(random)
                attempt++
            }
            problems.add(candidate)
        }
        return problems
    }

    private fun makersFor(grade: Grade): List<(Random) -> Problem> = when (grade) {
        Grade.FIRST -> listOf(::addWithin20, ::subWithin20)
        Grade.SECOND -> listOf(::addWithin100, ::subWithin100, ::timesTable, ::timesTableDiv)
        Grade.THIRD -> listOf(::addThreeDigit, ::subThreeDigit, ::mulTwoByOne, ::divTwoByOne)
        Grade.FOURTH -> listOf(::addWithin10000, ::subWithin10000, ::mulThreeByTwo, ::divThreeByTwo)
        Grade.FIFTH -> listOf(::decimalAdd, ::decimalSub, ::decimalMul, ::decimalDiv)
        Grade.SIXTH -> listOf(::fractionAdd, ::fractionSub, ::fractionMul, ::fractionDiv, ::percentOf)
    }

    // ---------------- 一年级：20 以内加减法 ----------------

    /** 20 以内加法：先定和 S，再拆成 `a + (S-a)`，避免"随机两个数再判和是否超 20"的筛法。 */
    private fun addWithin20(r: Random): Problem {
        val sum = r.nextInt(2, 21)
        val a = r.nextInt(1, sum)
        return Problem("$a + ${sum - a}", sum)
    }

    /** 20 以内减法：先定差，再取减数使被减数 ≤ 20，差天然非负。 */
    private fun subWithin20(r: Random): Problem {
        val diff = r.nextInt(1, 20)
        val b = r.nextInt(1, 21 - diff)
        return Problem("${diff + b} - $b", diff)
    }

    // ---------------- 二年级：100 以内加减法 + 表内乘除法 ----------------

    private fun addWithin100(r: Random): Problem {
        val sum = r.nextInt(11, 101)
        // 两个加数都 ≥ 2：避免出现 "+1" 这种没有训练价值的题面
        val a = r.nextInt(2, sum - 1)
        return Problem("$a + ${sum - a}", sum)
    }

    private fun subWithin100(r: Random): Problem {
        val diff = r.nextInt(2, 91)
        val b = r.nextInt(2, 101 - diff)
        return Problem("${diff + b} - $b", diff)
    }

    /** 表内乘法：两个因数都取 2..9，排除 ×1（无训练价值）与 ×0。 */
    private fun timesTable(r: Random): Problem {
        val a = r.nextInt(2, 10)
        val b = r.nextInt(2, 10)
        return Problem("$a × $b", a * b)
    }

    /** 表内除法就是表内乘法的逆：先定商与除数，被除数 = 商×除数，必然整除。 */
    private fun timesTableDiv(r: Random): Problem {
        val q = r.nextInt(2, 10)
        val b = r.nextInt(2, 10)
        return Problem("${q * b} ÷ $b", q)
    }

    // ---------------- 三年级：三位数加减 + 两位数乘除一位数 ----------------

    /** 三位数加法：两个加数都限定三位，和 ≤ 999（上界 999-a 保证 b 仍取得到三位数）。 */
    private fun addThreeDigit(r: Random): Problem {
        val a = r.nextInt(100, 900)
        val b = r.nextInt(100, 1000 - a)
        return Problem("$a + $b", a + b)
    }

    /** 三位数减法：被减数三位，减数三位，差 ≥ 1。 */
    private fun subThreeDigit(r: Random): Problem {
        val a = r.nextInt(200, 1000)
        val b = r.nextInt(100, a)
        return Problem("$a - $b", a - b)
    }

    /** 两位数乘一位数：99×9=891 ≤ 999 恒成立，无需额外约束。 */
    private fun mulTwoByOne(r: Random): Problem {
        val a = r.nextInt(10, 100)
        val b = r.nextInt(2, 10)
        return Problem("$a × $b", a * b)
    }

    /** 两位数除一位数：商的下界取 ceil(10/b) 保证被除数仍是两位数，上界取 floor(99/b)。 */
    private fun divTwoByOne(r: Random): Problem {
        val b = r.nextInt(2, 10)
        val lo = (10 + b - 1) / b
        val q = r.nextInt(lo, 99 / b + 1)
        return Problem("${q * b} ÷ $b", q)
    }

    // ---------------- 四年级：万以内加减 + 三位数乘除两位数 ----------------

    private fun addWithin10000(r: Random): Problem {
        val sum = r.nextInt(1000, 10000)
        val a = r.nextInt(100, sum - 99)
        return Problem("$a + ${sum - a}", sum)
    }

    private fun subWithin10000(r: Random): Problem {
        val a = r.nextInt(1000, 10000)
        val b = r.nextInt(100, a - 99)
        return Problem("$a - $b", a - b)
    }

    /**
     * 三位数乘两位数：先定两位数因数 b，再让 a 不超过 9999/b。
     * 9999/99 = 101 ≥ 100，所以 a 的取值区间恒非空，无需重试。
     *
     * 注意这个题型的积**必然 ≥ 1000**（100×10 就是 1000），也就是恒出 4 位数答案 —— 而
     * 4 位整串实测量识别全对率只有 87%。四年级是全功能唯一稳定踩在识别悬崖上的年级，
     * 界面上的"两次点击确认"正是为此设计的主要防线。
     */
    private fun mulThreeByTwo(r: Random): Problem {
        val b = r.nextInt(10, 100)
        val a = r.nextInt(100, minOf(999, MAX_ANSWER / b) + 1)
        return Problem("$a × $b", a * b)
    }

    /** 三位数除两位数：与两位数除一位数同理，下界保证被除数仍是三位数。 */
    private fun divThreeByTwo(r: Random): Problem {
        val b = r.nextInt(10, 100)
        val lo = (100 + b - 1) / b
        val q = r.nextInt(lo, 999 / b + 1)
        return Problem("${q * b} ÷ $b", q)
    }

    // ---------------- 五年级：小数四则（答案构造为整数） ----------------

    /**
     * 把 `scaled / 10^decimals` 渲染成小数文本，去掉小数末尾多余的 0（1.50 → "1.5"，10 → "1"）。
     *
     * 全程整数运算：题面必须是精确的，用 Double 会在 0.1+0.2 这类地方引入误差，
     * 而单元测试正是靠解析这个文本反推数值来验算的。
     */
    private fun decimalText(scaled: Int, decimals: Int): String {
        var scale = 1
        repeat(decimals) { scale *= 10 }
        val whole = scaled / scale
        val frac = scaled % scale
        if (frac == 0) return "$whole"
        return "$whole.${frac.toString().padStart(decimals, '0').trimEnd('0')}"
    }

    /**
     * 小数加法：先定整数答案 R，再把 R 拆成 `aInt.f + bInt.(10-f)` —— 小数部分互补成 1 产生进位，
     * 整数部分之和为 R-1，合起来恰好是 R。f ∈ 1..9 保证两个数都真的带小数位。
     */
    private fun decimalAdd(r: Random): Problem {
        val answer = r.nextInt(5, 100)
        val aInt = r.nextInt(1, answer - 1)
        val f = r.nextInt(1, 10)
        val a = aInt * 10 + f
        val b = (answer - aInt - 1) * 10 + (10 - f)
        return Problem("${decimalText(a, 1)} + ${decimalText(b, 1)}", answer)
    }

    /**
     * 小数减法：两个操作数**共用同一个小数位** f，小数部分相减抵消，差就是整数 R。
     *
     * 这不是偷懒，而是数学上的必然：设 a = A + f/10、b = B + g/10，a-b 为整数要求 (f-g)/10
     * 为整数，而 f-g ∈ (-9, 9)，故 f-g 只能等于 0。也就是说"答案必须是整数"的约束下，
     * `5.2 - 1.4` 这类需要退位的小数减法**无解**。换来的教学点是"小数部分相同、相减抵消"。
     */
    private fun decimalSub(r: Random): Problem {
        val answer = r.nextInt(2, 100)
        val bInt = r.nextInt(1, 21)
        val f = r.nextInt(1, 10)
        val a = (answer + bInt) * 10 + f
        val b = bInt * 10 + f
        return Problem("${decimalText(a, 1)} - ${decimalText(b, 1)}", answer)
    }

    /**
     * 小数乘法。一位小数 × 整数要得到整数，末位 f 与乘数 b 必须满足 `f*b ≡ 0 (mod 10)`；
     * 枚举后只剩两个解族，直接按解族构造，不做随机筛选：
     * - `i.5 × {2,4,6,8}`（f=5 配偶数）
     * - `X.25/.5/.75 × {4,8}`（两位小数的尾数配 4 或 8）
     */
    private fun decimalMul(r: Random): Problem {
        if (r.nextBoolean()) {
            val a = r.nextInt(1, 10) * 10 + 5
            val b = EVEN_MULTIPLIERS.random(r)
            return Problem("${decimalText(a, 1)} × $b", a * b / 10)
        }
        val whole = r.nextInt(1, 10)
        val (fracScaled, mults) = HUNDREDTHS_FAMILIES.random(r)
        val a = whole * 100 + fracScaled
        val b = mults.random(r)
        return Problem("${decimalText(a, 2)} × $b", a * b / 100)
    }

    /**
     * 小数除法。两条解族都是"先定商"：
     * - D1 一位小数 ÷ 一位小数：除数取与 10 互质的 {3,7,9}，商的末位避开 0 —— 这样被除数的
     *   末位必然非 0，不会退化成 `2.0 ÷ 0.3` 这种名不副实的整数除法；商 ≥ 4 保证被除数 ≥ 12，
     *   不会写成 `0.3` 这种整数部分为 0 的形式。两个约束都是构造性满足的，无需重试。
     * - D2 整数 ÷ 一位小数：被除数 = 商·b/10 要为整数，即 `商·b ≡ 0 (mod 10)`。b 含因子 2 时
     *   让商取 5 的倍数，b = 5 时让商取偶数，分开构造，同样无筛除。
     */
    private fun decimalDiv(r: Random): Problem {
        if (r.nextBoolean()) {
            val b = COPRIME_TO_TEN.random(r)
            val raw = r.nextInt(4, 40)
            val answer = if (raw % 10 == 0) raw - 1 else raw
            val a = answer * b
            return Problem("${decimalText(a, 1)} ÷ ${decimalText(b, 1)}", answer)
        }
        val b = INTEGER_DIVISORS.random(r)
        val step = if (b == 5) 2 else 5
        val answer = r.nextInt(2, 99 / step + 1) * step
        return Problem("${answer * b / 10} ÷ ${decimalText(b, 1)}", answer)
    }

    // ---------------- 六年级：分数四则 + 百分数（答案构造为整数） ----------------

    /**
     * 分数加法：同分母真分数 `p/D + (D-p)/D`，和为 1。
     *
     * 答案恒为 1 同样是数学必然，不是设计得不好：设 p+q = R·D 且 p,q ≤ D，则 R·D ≤ 2D，
     * 即 R ≤ 2；R = 2 只能退化成 `1 + 1`（假分数），无练习价值。所以只剩 R = 1。
     * 变化性来自分母与拆分点的组合，以及**约分后显示** —— D=8、p=2 正好渲染成课本里
     * 最常见的 `1/4 + 3/4`。
     */
    private fun fractionAdd(r: Random): Problem {
        val d = r.nextInt(2, MAX_DENOMINATOR + 1)
        val p = r.nextInt(1, d)
        return Problem("${reduced(p, d)} + ${reduced(d - p, d)}", 1)
    }

    /**
     * 分数减法：被减数取假分数 `(R·D+B)/D`，减数取真分数 `B/D`，差恰为整数 R。
     *
     * 被减数为什么必须是假分数：真分数同分母相减的差恒 < 1，而答案 0 已被排除（画布上单独写
     * 一个 "0" 是闭合环，与"什么都没写"难以区分），所以真分数减真分数在此约束下无解。
     */
    private fun fractionSub(r: Random): Problem {
        val d = r.nextInt(2, MAX_DENOMINATOR + 1)
        val answer = r.nextInt(1, 5)
        val b = r.nextInt(1, d)
        val a = answer * d + b
        return Problem("$a/$d - $b/$d", answer)
    }

    /**
     * 分数乘法：`(p/D) × m` 要整除，等价于 m 含因子 D/gcd(p,D)。
     * 约简后令 m = (D/g)·t，此时答案 = (p/g)·t 必然是整数 —— 全程无筛除。
     * 例：D=4、p=3、t=2 → `3/4 × 8 = 6`。
     */
    private fun fractionMul(r: Random): Problem {
        val d = r.nextInt(2, MAX_DENOMINATOR + 1)
        val p = r.nextInt(1, d)
        val g = gcd(p, d)
        val den = d / g
        val p2 = p / g
        // 整数因数 m = den·t，上限取 24 而非 12：这样 den 取到最大（12）时 t 仍能到 2。
        // 原因见下面 minT —— 若 t 只能取 1，p2 = 1 的题面就退化成"1/7 × 7 = 1"。
        val maxT = minOf(4, 24 / den)
        // 约分后分子为 1（即 p 整除 d）时强制 t ≥ 2，避开"分数乘以自己的分母"这种答案恒为 1
        // 的退化题 —— 六年级已经有一整个"分数加法答案恒为 1"的题型了，再叠加会让整组题过于单调。
        // 24/den ≥ 2 对 den ≤ 12 恒成立，所以 minT ≤ maxT 恒成立，无需重试。
        val minT = if (p2 == 1) 2 else 1
        val t = r.nextInt(minT, maxT + 1)
        return Problem("${reduced(p, d)} × ${den * t}", p2 * t)
    }

    /**
     * 分数除法：`a ÷ (p'/n') = a·n'/p'` 要整除，因 p' 与 n' 互质，等价于 p' | a。
     * 令 a = p'·t，则答案 = n'·t 必然是整数 —— 同样无筛除。
     * 例：p'=1、n'=4、t=2 → `2 ÷ 1/4 = 8`；p'=3、n'=4、t=2 → `6 ÷ 3/4 = 8`。
     * 约分后 p'=1 的情况很常见（`2 ÷ 1/4`），这是课本里的标准题型，刻意保留。
     */
    private fun fractionDiv(r: Random): Problem {
        val n = r.nextInt(2, MAX_DENOMINATOR + 1)
        val p = r.nextInt(1, n)
        val g = gcd(p, n)
        val p2 = p / g
        val n2 = n / g
        val t = r.nextInt(1, minOf(99 / p2, 99 / n2) + 1)
        return Problem("${p2 * t} ÷ $p2/$n2", n2 * t)
    }

    /**
     * 百分数：先定百分比 P，再定答案 R，反推整体 N = R·100/P。
     * P 只取 {5,10,20,25,50} —— 这些值能让 100/P 为整数，换任何别的 P 都会出现 N 非整数的情况。
     * N 只是题面不参与手写，所以它大一些不影响识别；这里仍限制 N ≤ 999 以便阅读。
     */
    private fun percentOf(r: Random): Problem {
        val p = PERCENTS.random(r)
        val answerMax = minOf(99, 999 * p / 100)
        val answer = r.nextInt(1, answerMax + 1)
        return Problem("${answer * 100 / p} 的 $p%", answer)
    }

    // ---------------- 工具 ----------------

    /** 约分后渲染成分数文本。 */
    private fun reduced(num: Int, den: Int): String {
        val g = gcd(num, den)
        return "${num / g}/${den / g}"
    }

    private tailrec fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

    /** `i.5 × 偶数` 必然为整数（末位 5 乘偶数得整十）。 */
    private val EVEN_MULTIPLIERS = listOf(2, 4, 6, 8)

    /** `X.25/.5/.75` 配 4 或 8，乘积的末两位必然是 100 的倍数。 */
    private val HUNDREDTHS_FAMILIES = listOf(
        25 to listOf(4, 8),
        50 to EVEN_MULTIPLIERS,
        75 to listOf(4, 8)
    )

    /** 与 10 互质的除数：商只要末位非 0，被除数的末位就必然非 0。 */
    private val COPRIME_TO_TEN = listOf(3, 7, 9)

    /** 整数 ÷ 一位小数的可选除数。 */
    private val INTEGER_DIVISORS = listOf(2, 4, 5, 6, 8)

    /** 能让 `R·100/P` 恒为整数的百分比。 */
    private val PERCENTS = listOf(5, 10, 20, 25, 50)
}
