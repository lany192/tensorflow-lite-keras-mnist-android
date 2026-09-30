package com.github.lany192.mnist.practice

import com.github.lany192.mnist.recognize.MainActivity
import kotlin.random.Random

/**
 * 小学一~六年级口算题的生成器。
 *
 * 三条贯穿全文的约束：
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
 * 3. **二年级及以上有组合运算，三年级及以上有带小括号的组合运算**。组合运算题面含两个运算符，
 *    按小学的"先乘除后加减"求值（`×`、`÷`、`的` 同级且优先于 `+`、`-`），括号内优先。
 *    这条语义由测试里那个独立的递归下降求值器独立验算 —— 生成器只负责构造，不负责解释。
 *    改题库前先读 `MathProblemGeneratorTest`，那里同时钉住了整数答案、非负中间值、整除、
 *    各年级数值域和"小括号只从三年级开始出现"。
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

    /**
     * 各年级的题型清单。
     *
     * **组合运算的 maker 一律追加在末尾**：轮转是按 maker 逐个取的，往末尾追加不改变既有题型的
     * 出现次数，`gradeFour_producesFourDigitAnswers` 这类"某些题型必须出现"的断言才不会因为
     * 分母变大而失效。反过来，把新题型插在中间会让既有题型的轮转次数整体错位。
     */
    private fun makersFor(grade: Grade): List<(Random) -> Problem> = when (grade) {
        Grade.FIRST -> listOf(::addWithin20, ::subWithin20)
        Grade.SECOND -> listOf(
            ::addWithin100, ::subWithin100, ::timesTable, ::timesTableDiv,
            ::mixedWithin100
        )

        Grade.THIRD -> listOf(
            ::addThreeDigit, ::subThreeDigit, ::mulTwoByOne, ::divTwoByOne,
            ::mixedTwoDigit, ::parenTwoDigit
        )

        Grade.FOURTH -> listOf(
            ::addWithin10000, ::subWithin10000, ::mulThreeByTwo, ::divThreeByTwo,
            ::mixedThreeDigit, ::parenThreeDigit
        )

        Grade.FIFTH -> listOf(
            ::decimalAdd, ::decimalSub, ::decimalMul, ::decimalDiv,
            ::mixedDecimal, ::parenDecimal
        )

        Grade.SIXTH -> listOf(
            ::fractionAdd, ::fractionSub, ::fractionMul, ::fractionDiv, ::percentOf,
            ::mixedFraction, ::parenFraction
        )
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

    /**
     * 二下 组合运算：`a × b ± c`（表内乘法）或 `a ÷ b ± c`（表内除法），答案仍在 100 以内。
     *
     * 商的下界取 3 而不是 2 是这个题型唯一需要解释的地方：减法分支的减数取 `2..商-1`，
     * 商若为 2 该区间为空。这里用**抬高下界**来构造性满足，而不是生成后再筛掉。
     */
    private fun mixedWithin100(r: Random): Problem = if (r.nextBoolean()) {
        val a = r.nextInt(2, 10)
        val b = r.nextInt(2, 10)
        withAddend("$a × $b", a * b, SUMMAND_MIN, 100, r)
    } else {
        val b = r.nextInt(2, 10)
        val q = r.nextInt(3, 10)
        withAddend("${q * b} ÷ $b", q, SUMMAND_MIN, 100, r)
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

    /**
     * 三上 组合运算：`a × b ± c`（两位数 × 一位数）或 `a ÷ b ± c`（两位数 ÷ 一位数），答案 ≤ 999。
     *
     * 除法的商下界比 [divTwoByOne] 多抬了一档到 3，理由与二年级相同：减法分支的减数取 `2..商-1`。
     */
    private fun mixedTwoDigit(r: Random): Problem = if (r.nextBoolean()) {
        val a = r.nextInt(10, 100)
        val b = r.nextInt(2, 10)
        withAddend("$a × $b", a * b, SUMMAND_MIN_LARGE, THREE_DIGIT_MAX, r)
    } else {
        val b = r.nextInt(2, 10)
        val lo = maxOf(3, (10 + b - 1) / b)
        val q = r.nextInt(lo, 99 / b + 1)
        withAddend("${q * b} ÷ $b", q, SUMMAND_MIN_LARGE, THREE_DIGIT_MAX, r)
    }

    /**
     * 三下 带小括号的组合运算：`(a ± b) × c` 或 `(a ± b) ÷ c`，括号内两个加数都是两位数、c 是一位数。
     *
     * 括号内的值先定，再拆成两个两位数：
     * - 加法把和限制在 `[20, 198]`（两个两位数的最小/最大和），再保证拆出的 a、b 仍落在两位数区间；
     * - 减法先定差 `d ∈ [2, 89]`，再由 `b ∈ [10, 100-d)` 反推 `a = b + d`，这样两个数都必然是两位数；
     * - 除法把括号内的值构造成 `q·c`，整除是构造性的，不靠筛除。
     *
     * 乘法的和上界额外套了 `999 / c`，使答案不越过三年级的三位数上限。
     */
    private fun parenTwoDigit(r: Random): Problem {
        val c = r.nextInt(2, 10)
        if (r.nextBoolean()) {
            return if (r.nextBoolean()) {
                val sum = r.nextInt(20, minOf(198, THREE_DIGIT_MAX / c) + 1)
                val a = r.nextInt(maxOf(10, sum - 99), minOf(99, sum - 10) + 1)
                Problem("($a + ${sum - a}) × $c", sum * c)
            } else {
                val diff = r.nextInt(2, 90)
                val b = r.nextInt(10, 100 - diff)
                Problem("(${b + diff} - $b) × $c", diff * c)
            }
        }
        return if (r.nextBoolean()) {
            val q = r.nextInt((20 + c - 1) / c, 198 / c + 1)
            val sum = q * c
            val a = r.nextInt(maxOf(10, sum - 99), minOf(99, sum - 10) + 1)
            Problem("($a + ${sum - a}) ÷ $c", q)
        } else {
            val q = r.nextInt(2, 89 / c + 1)
            val diff = q * c
            val b = r.nextInt(10, 100 - diff)
            Problem("(${b + diff} - $b) ÷ $c", q)
        }
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

    /**
     * 四上 组合运算：`a × b ± c`（三位数 × 两位数）或 `a ÷ b ± c`（三位数 ÷ 两位数），答案 ≤ 9999。
     *
     * 乘法沿用了 [mulThreeByTwo] 的 `MAX_ANSWER / b` 上界；`b = 99` 时 a 的取值区间退化成 `{100}`，
     * 但区间仍非空，且此时乘积 9900 离 9999 还留有 99 的余量，加减项依然取得到。
     */
    private fun mixedThreeDigit(r: Random): Problem = if (r.nextBoolean()) {
        val b = r.nextInt(10, 100)
        // 上界与 withAddend 的加数下限绑定：留够 SUMMAND_MIN_LARGE，加法分支的区间才不会为空
        val a = r.nextInt(100, minOf(999, (MAX_ANSWER - SUMMAND_MIN_LARGE) / b) + 1)
        withAddend("$a × $b", a * b, SUMMAND_MIN_LARGE, MAX_ANSWER, r)
    } else {
        val b = r.nextInt(10, 100)
        val lo = maxOf(3, (100 + b - 1) / b)
        val q = r.nextInt(lo, 999 / b + 1)
        withAddend("${q * b} ÷ $b", q, SUMMAND_MIN_LARGE, MAX_ANSWER, r)
    }

    /**
     * 四下 带小括号的组合运算：`(a ± b) × c` 或 `(a ± b) ÷ c`，括号内是三位数、c 是一位数。
     *
     * 构造与 [parenTwoDigit] 同构，只把两位数区间放大到三位数。乘法的被乘数是括号内的和/差，
     * 所以答案上界由 `MAX_ANSWER / c` 控制，而不是由两个三位数本身控制。
     */
    private fun parenThreeDigit(r: Random): Problem {
        val c = r.nextInt(2, 10)
        if (r.nextBoolean()) {
            return if (r.nextBoolean()) {
                val sum = r.nextInt(200, minOf(1998, MAX_ANSWER / c) + 1)
                val a = r.nextInt(maxOf(100, sum - 999), minOf(999, sum - 100) + 1)
                Problem("($a + ${sum - a}) × $c", sum * c)
            } else {
                val diff = r.nextInt(2, 900)
                val b = r.nextInt(100, 1000 - diff)
                Problem("(${b + diff} - $b) × $c", diff * c)
            }
        }
        return if (r.nextBoolean()) {
            val q = r.nextInt((200 + c - 1) / c, 1998 / c + 1)
            val sum = q * c
            val a = r.nextInt(maxOf(100, sum - 999), minOf(999, sum - 100) + 1)
            Problem("($a + ${sum - a}) ÷ $c", q)
        } else {
            val q = r.nextInt(2, 899 / c + 1)
            val diff = q * c
            val b = r.nextInt(100, 1000 - diff)
            Problem("(${b + diff} - $b) ÷ $c", q)
        }
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

    /**
     * 五上 组合运算：`a × b ± c` 或 `a ÷ b ± c`，题面仍含小数点、答案仍是整数。
     *
     * 乘法分支的关键是**十分位取余**：一位小数 `a = (10A+f)/10` 乘整数 b 后，
     * 积的十分位值 `scaled = a·b`，小数部分只剩 `g = scaled % 10`。加数的小数部分取 `(10-g) % 10`
     * 与它**互补**、减数取 `g` 与它**相消**，整体因此被凑成整数。
     * 全程整数运算，一个浮点数都不出现 —— 与 [decimalText] 的理由相同。
     *
     * 除法分支直接复用 [decimalDiv] 的第一条解族（商恒为整数），后面接一个整数加减项。
     */
    private fun mixedDecimal(r: Random): Problem {
        if (r.nextBoolean()) {
            val b = COPRIME_TO_TEN.random(r)
            val raw = r.nextInt(4, 40)
            val q = if (raw % 10 == 0) raw - 1 else raw
            return withAddend("${decimalText(q * b, 1)} ÷ ${decimalText(b, 1)}", q, SUMMAND_MIN, MAX_ANSWER, r)
        }
        val a = r.nextInt(1, 100) * 10 + r.nextInt(1, 10)
        val b = r.nextInt(2, 10)
        val scaled = a * b
        val whole = scaled / 10
        val g = scaled % 10
        if (r.nextBoolean()) {
            val frac = (10 - g) % 10
            // frac 为 0 时加数退化成整数，此时整数部分从 2 起步，避开 "+ 1" 这种没有训练价值的题面
            val head = r.nextInt(if (frac == 0) 2 else 1, 100)
            return Problem(
                "${decimalText(a, 1)} × $b + ${decimalText(head * 10 + frac, 1)}",
                whole + head + if (g == 0) 0 else 1
            )
        }
        // 减数的小数部分与积相同，相减抵消；整数部分上界由 whole - 1 给出，保证答案 ≥ 1
        val head = r.nextInt(if (g == 0 && whole > 2) 2 else 1, whole)
        return Problem(
            "${decimalText(a, 1)} × $b - ${decimalText(head * 10 + g, 1)}",
            whole - head
        )
    }

    /**
     * 五下 带小括号的组合运算：`(a ± b) × c` 或 `(a ± b) ÷ c`，括号内是两个一位小数、c 是一位数。
     *
     * 括号内的值先定成整数 [value]，再按 [splitDecimal] 拆成两个小数：加法用互补的小数部分、
     * 减法用相同的小数部分，括号内必然是整数。除法分支把 [value] 构造成 c 的倍数，整除同样是构造性的。
     */
    private fun parenDecimal(r: Random): Problem {
        val c = r.nextInt(2, 10)
        val f = r.nextInt(1, 10)
        val adding = r.nextBoolean()
        if (r.nextBoolean()) {
            val value = if (adding) {
                r.nextInt(3, minOf(199, MAX_ANSWER / c) + 1)
            } else {
                r.nextInt(2, minOf(99, MAX_ANSWER / c) + 1)
            }
            val (left, right) = splitDecimal(value, f, adding, r)
            val op = if (adding) "+" else "-"
            return Problem("(${decimalText(left, 1)} $op ${decimalText(right, 1)}) × $c", value * c)
        }
        val q = if (adding) r.nextInt(2, 199 / c + 1) else r.nextInt(2, 99 / c + 1)
        val value = q * c
        val (left, right) = splitDecimal(value, f, adding, r)
        val op = if (adding) "+" else "-"
        return Problem("(${decimalText(left, 1)} $op ${decimalText(right, 1)}) ÷ $c", q)
    }

    /**
     * 把括号内的整数值 [value] 拆成两个一位小数，返回它们的**十分位**值，交给 [decimalText] 渲染。
     *
     * - 加法：小数部分取 `f` 与 `10-f`，相加恰好进位 1，于是 `a + b = value`；
     * - 减法：两个数共用小数部分 `f`，相减抵消，于是 `a - b = value`。
     *
     * 这就是这个题型不需要任何筛除的原因 —— 括号内是不是整数完全由构造决定。
     * `f ∈ 1..9` 保证两个数都真的带小数位；拆出的两个整数部分都 ≥ 1，不会写出 `0.5` 这种形式。
     */
    private fun splitDecimal(value: Int, f: Int, adding: Boolean, r: Random): Pair<Int, Int> =
        if (adding) {
            val left = r.nextInt(1, value - 1)
            (left * 10 + f) to ((value - 1 - left) * 10 + (10 - f))
        } else {
            val right = r.nextInt(1, 20)
            ((right + value) * 10 + f) to (right * 10 + f)
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

    /**
     * 六上 组合运算：`a × b ± c`（分数 × 整数）或 `a ÷ b ± c`（整数 ÷ 分数），答案 ≤ 9999。
     *
     * 乘除部分原样复用 [fractionMul] / [fractionDiv] 的 gcd 构造 —— 那两条构造本来就把结果
     * 保证成非负整数，这里只是在后面接一个整数加减项。既有的 `p2 == 1 时强制 t ≥ 2` 也一并保留，
     * 否则会出现"1/7 × 7 = 1"这种答案恒为 1 的退化题。
     */
    private fun mixedFraction(r: Random): Problem {
        if (r.nextBoolean()) {
            val d = r.nextInt(2, MAX_DENOMINATOR + 1)
            val p = r.nextInt(1, d)
            val g = gcd(p, d)
            val den = d / g
            val p2 = p / g
            val t = r.nextInt(if (p2 == 1) 2 else 1, minOf(4, 24 / den) + 1)
            return withAddend("${reduced(p, d)} × ${den * t}", p2 * t, SUMMAND_MIN, MAX_ANSWER, r)
        }
        val n = r.nextInt(2, MAX_DENOMINATOR + 1)
        val p = r.nextInt(1, n)
        val g = gcd(p, n)
        val p2 = p / g
        val n2 = n / g
        val t = r.nextInt(1, minOf(99 / p2, 99 / n2) + 1)
        return withAddend("${p2 * t} ÷ $p2/$n2", n2 * t, SUMMAND_MIN, MAX_ANSWER, r)
    }

    /**
     * 六下 带小括号的组合运算：`(p/d ± q/d) × m`。
     *
     * `(p/d ± q/d) × m` 要是整数，等价于 `d | (p±q)·m`。令 `u = p ± q`、`g = gcd(u, d)`，
     * 则 m 取 `(d/g)` 的倍数即可 —— 和 [fractionMul] 是同一把钥匙，全程无筛除。
     *
     * 额外要求 `u < d`（括号内是真分数且不等于 1）：这保证 `d/g ≥ 2`，从而乘数 `m ≥ 2`，
     * 不会退化成"乘以 1"这种没有训练价值的题面；减法分支还要求 `u ≥ 2`，避开"差为 1/d"。
     */
    private fun parenFraction(r: Random): Problem {
        val d = r.nextInt(4, MAX_DENOMINATOR + 1)
        val adding = r.nextBoolean()
        val u: Int
        val left: Int
        val right: Int
        if (adding) {
            u = r.nextInt(2, d)
            left = r.nextInt(1, u)
            right = u - left
        } else {
            u = r.nextInt(2, d - 1)
            right = r.nextInt(1, d - u)
            left = right + u
        }
        val g = gcd(u, d)
        val den = d / g
        val t = r.nextInt(1, 99 / den + 1)
        val op = if (adding) "+" else "-"
        return Problem("(${reduced(left, d)} $op ${reduced(right, d)}) × ${den * t}", u / g * t)
    }

    // ---------------- 工具 ----------------

    /**
     * 给一个已经算好的乘积或商接上加减项，构成组合运算题面。
     *
     * **加减项有一半概率写在前面**，这不是排版偏好：`a × b + c` 从左到右算也能得到同一个答案，
     * 学生不用"先乘除后加减"也蒙不丢；只有 `c + a × b` 这个形态才真的要求那条规则。
     * 两种写法都要有，否则这个题型就落了空。
     *
     * 加减项还不得比乘除部分大（见 [addendUpperBound]）：`81 ÷ 9 + 87` 这种题面里乘除部分被淹没，
     * 两步运算只剩一步是有效的。
     *
     * 四个分支的取值区间都靠构造保证非空，不需要任何重试：
     * - `base` 小于 3 时（六年级分数乘法的最小积是 2）减法分支无解，自动只出加法题，
     *   而不是让 `nextInt(2, 2)` 抛异常；
     * - 其余分支由各调用点的 `cap - base ≥ minAddend` 与 [addendUpperBound] ≥ minAddend
     *   两条不变量保证区间宽度不为 0。
     *
     * @param baseText 乘除部分的题面，如 `"25 × 3"`
     * @param base 乘除部分的数值，也就是答案里不含加减项的那一半
     * @param minAddend 加减项下限，见 [SUMMAND_MIN] 与 [SUMMAND_MIN_LARGE]
     * @param cap 该年级的答案上限
     */
    private fun withAddend(baseText: String, base: Int, minAddend: Int, cap: Int, r: Random): Problem {
        val adding = base < 3 || r.nextBoolean()
        val addendFirst = r.nextBoolean()
        return when {
            adding && addendFirst -> {
                val c = r.nextInt(minAddend, addendUpperBound(base, minAddend, cap) + 1)
                Problem("$c + $baseText", c + base)
            }

            adding -> {
                val c = r.nextInt(minAddend, addendUpperBound(base, minAddend, cap) + 1)
                Problem("$baseText + $c", base + c)
            }

            addendFirst -> {
                // 被减数同样按 addendUpperBound 限制，使差与乘除部分量级相当：
                // `9164 - 4/11 × 22` 这种题面里，学生实际只练到了一次四位数减法
                val c = r.nextInt(base + minAddend, minOf(cap, base + addendUpperBound(base, minAddend, cap)) + 1)
                Problem("$c - $baseText", c - base)
            }

            else -> {
                val c = r.nextInt(2, base)
                Problem("$baseText - $c", base - c)
            }
        }
    }

    /**
     * 加法分支里加减项的上界：既不越过该年级的答案上限，也不该比乘除部分本身还大
     * —— 否则学生实际只练到了加法。
     *
     * 乘除部分很小时以 [minAddend] 为准，这样上界永远 ≥ 下限、区间不会塌成空的：
     * `cap - base` 由各调用点保证 ≥ [minAddend]，`maxOf` 那一项也恒 ≥ [minAddend]。
     */
    private fun addendUpperBound(base: Int, minAddend: Int, cap: Int): Int =
        minOf(cap - base, maxOf(base, minAddend))

    /** 约分后渲染成分数文本。 */
    private fun reduced(num: Int, den: Int): String {
        val g = gcd(num, den)
        return "${num / g}/${den / g}"
    }

    private tailrec fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

    /** 低年级加减项的下限。取 2 而不是 1，避开 `+ 1` 这种没有训练价值的题面。 */
    private const val SUMMAND_MIN = 2

    /** 中高年级加减项的下限。从两位数起步，否则组合运算和单运算符题的难度没有区别。 */
    private const val SUMMAND_MIN_LARGE = 10

    /** 三年级的三位数答案上限。 */
    private const val THREE_DIGIT_MAX = 999

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
