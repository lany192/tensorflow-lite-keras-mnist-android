package com.github.lany192.mnist.practice

import kotlin.math.abs
import kotlin.random.Random
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 出题逻辑的纯 JVM 测试。
 *
 * 核心是下面这个**测试专用的独立求值器**：它不复用生成器的任何代码，而是把 `Problem.expression`
 * 字符串重新解析回数值，再断言等于 `Problem.answer`。
 *
 * 它唯一能抓出、也最可能真实发生的那类 bug 是**题面与答案字段不一致**（比如手滑写成
 * `Problem("$a × $b + $c", a * b - c)`）。纯不变量测试（答案非负、≤9999、不重复）对这类错误
 * 完全无感，会全绿放行，所以这段解析代码不能省。
 *
 * 求值器按小学的运算顺序实现：**先乘除后加减**，`的` 与乘除同级，小括号优先。
 * 组合运算题面（`25 × 3 + 48`、`(25 + 15) × 3`）必须靠这条规则才能算对 —— 一个按从左到右
 * 求值的坏求值器会把 `2 + 3 × 4` 算成 20，从而把所有组合运算题都判成"题面与答案不一致"，
 * 或者反过来把生成器的真实 bug 掩盖过去。所以 [evaluator_computesEverySyntax] 里专门放了
 * 优先级和小括号的自测。
 */

/**
 * 精确有理数。
 *
 * **绝不能用 Double 代替**：小数题在浮点下会出现 0.1+0.2 这类误差，既可能把正确的题判错，
 * 也可能把 `3.5 + 0.5` 这种边界掩盖成"恰好相等"，让真正的不一致漏网。
 */
private class Fraction private constructor(val numerator: Long, val denominator: Long) {
    val isInteger: Boolean get() = denominator == 1L
    val longValue: Long get() = numerator / denominator

    // 不用 data class：私有主构造器会让生成的 copy() 越过可见性，编译器已就此告警。
    // Fraction 恒为既约形式，所以分子分母分别相等就等价于数值相等。
    override fun equals(other: Any?): Boolean =
        other is Fraction && numerator == other.numerator && denominator == other.denominator

    override fun hashCode(): Int = 31 * numerator.hashCode() + denominator.hashCode()

    operator fun plus(other: Fraction) = of(
        numerator * other.denominator + other.numerator * denominator,
        denominator * other.denominator
    )

    operator fun minus(other: Fraction) = of(
        numerator * other.denominator - other.numerator * denominator,
        denominator * other.denominator
    )

    operator fun times(other: Fraction) = of(numerator * other.numerator, denominator * other.denominator)

    operator fun div(other: Fraction) = of(numerator * other.denominator, denominator * other.numerator)

    override fun toString(): String = if (isInteger) "$longValue" else "$numerator/$denominator"

    companion object {
        fun of(num: Long, den: Long = 1L): Fraction {
            require(den != 0L) { "分母不能为 0" }
            val sign = if (den < 0) -1L else 1L
            val divisor = gcd(abs(num), abs(den)).coerceAtLeast(1L)
            return Fraction(sign * num / divisor, sign * den / divisor)
        }

        private tailrec fun gcd(a: Long, b: Long): Long = if (b == 0L) a else gcd(b, a % b)
    }
}

/**
 * 递归下降的题面求值器，语法与小学的运算顺序一一对应：
 *
 * ```
 * expression := term (('+' | '-') term)*
 * term       := factor (('×' | '÷' | '的') factor)*
 * factor     := '(' expression ')' | operand
 * ```
 *
 * `的` 归到乘法这一级（`80 的 25%` = `80 × 25/100`），`%` 由 [parseOperand] 直接解析成百分数，
 * 于是百分数与组合运算能走同一条通路，不需要为它单开一个分支。
 *
 * 求值过程中顺带把两类**不变量违例**记进 [violations]：减法结果为负、除法不整除。
 * 合并进同一次遍历而不是再写一遍切段逻辑，是为了让"怎么算"只有一个真相来源 —— 两套遍历
 * 迟早会对同一条题面给出不同的解释。
 */
private class Evaluator(private val expression: String) {
    private val tokens: List<String> = expression
        .replace("(", " ( ")
        .replace(")", " ) ")
        .trim()
        .split(Regex("\\s+"))

    private var cursor = 0
    private val found = mutableListOf<String>()

    /** 题面的值。 */
    val value: Fraction = parseAdditive().also {
        assertEquals("题面有无法解析的多余内容：$expression", tokens.size, cursor)
    }

    /** 求值过程中发现的不变量违例，空列表表示全部满足。 */
    val violations: List<String> get() = found

    private fun parseAdditive(): Fraction {
        var acc = parseMultiplicative()
        while (cursor < tokens.size && (tokens[cursor] == "+" || tokens[cursor] == "-")) {
            val op = tokens[cursor++]
            val rhs = parseMultiplicative()
            if (op == "-") {
                // Fraction 恒把符号收敛到分子上，所以分子为负就是"减法出现负数"
                if ((acc - rhs).numerator < 0) found += "减法出现负数：$expression 中 $acc - $rhs"
                acc -= rhs
            } else {
                acc += rhs
            }
        }
        return acc
    }

    private fun parseMultiplicative(): Fraction {
        var acc = parseFactor()
        while (cursor < tokens.size && tokens[cursor] in MULTIPLICATIVE) {
            val op = tokens[cursor++]
            val rhs = parseFactor()
            acc = if (op == "÷") {
                val quotient = acc / rhs
                if (!quotient.isInteger) found += "除法不整除：$expression 中 $acc ÷ $rhs = $quotient"
                quotient
            } else {
                acc * rhs
            }
        }
        return acc
    }

    private fun parseFactor(): Fraction = if (tokens[cursor] == "(") {
        cursor++
        val inner = parseAdditive()
        assertEquals("括号没有闭合：$expression", ")", tokens[cursor])
        cursor++
        inner
    } else {
        parseOperand(tokens[cursor++])
    }

    private companion object {
        val MULTIPLICATIVE = setOf("×", "÷", "的")
    }
}

private fun evaluate(expression: String): Fraction = Evaluator(expression).value

private fun violationsOf(expression: String): List<String> = Evaluator(expression).violations

/** 操作数：整数、一位/多位小数、分数、百分数各一种写法。 */
private fun parseOperand(token: String): Fraction = when {
    token.endsWith("%") -> Fraction.of(token.removeSuffix("%").toLong(), 100)
    "/" in token -> token.split("/").let { Fraction.of(it[0].toLong(), it[1].toLong()) }
    "." in token -> {
        val decimals = token.substringAfter(".").length
        var scale = 1L
        repeat(decimals) { scale *= 10 }
        Fraction.of(token.replace(".", "").toLong(), scale)
    }
    else -> Fraction.of(token.toLong())
}

class MathProblemGeneratorTest {

    // ---------------- 求值器自测：先证明这把尺子是准的 ----------------

    @Test
    fun evaluator_computesEverySyntax() {
        assertEquals(Fraction.of(68), evaluate("23 + 45"))
        assertEquals(Fraction.of(33), evaluate("80 - 47"))
        assertEquals(Fraction.of(81), evaluate("9 × 9"))
        assertEquals(Fraction.of(7), evaluate("56 ÷ 8"))
        assertEquals(Fraction.of(5), evaluate("3.5 + 1.5"))
        assertEquals(Fraction.of(7), evaluate("6.3 ÷ 0.9"))
        assertEquals(Fraction.of(1), evaluate("1/4 + 3/4"))
        assertEquals(Fraction.of(2), evaluate("7/3 - 1/3"))
        assertEquals(Fraction.of(6), evaluate("3/4 × 8"))
        assertEquals(Fraction.of(8), evaluate("2 ÷ 1/4"))
        assertEquals(Fraction.of(20), evaluate("80 的 25%"))
    }

    /**
     * 组合运算的求值必须遵守"先乘除后加减"，小括号必须优先。
     *
     * 这几条是这个文件里最要紧的自测：一个从左到右求值的坏求值器会把 `2 + 3 × 4` 算成 20，
     * 于是要么把所有组合运算题误判成"题面与答案不一致"，要么在生成器真的算错时与之抵消。
     */
    @Test
    fun evaluator_followsOperationOrderAndParentheses() {
        assertEquals(Fraction.of(14), evaluate("2 + 3 × 4"))
        assertEquals(Fraction.of(20), evaluate("(2 + 3) × 4"))
        assertEquals(Fraction.of(58), evaluate("100 - 6 × 7"))
        assertEquals(Fraction.of(24), evaluate("36 ÷ 4 + 15"))
        assertEquals(Fraction.of(9), evaluate("2.5 × 3 + 1.5"))
        assertEquals(Fraction.of(16), evaluate("(1.5 + 2.5) × 4"))
        assertEquals(Fraction.of(120), evaluate("(25 + 15) × 3"))
        assertEquals(Fraction.of(6), evaluate("(1/2 + 1/4) × 8"))
        assertEquals(Fraction.of(80), evaluate("200 的 25% + 30"))
    }

    /** 求值器必须能判错。一个恒返回期望值的坏求值器会让下面所有生成器测试变成空转。 */
    @Test
    fun evaluator_rejectsWrongExpectedValue() {
        assertNotEquals(Fraction.of(2), evaluate("1/4 + 3/4"))
        assertNotEquals(Fraction.of(5), evaluate("6.3 ÷ 0.9"))
        assertNotEquals(Fraction.of(45), evaluate("23 + 45"))
        // 20 是"从左到右"误算 `2 + 3 × 4` 的结果，必须被挡住
        assertNotEquals(Fraction.of(20), evaluate("2 + 3 × 4"))
    }

    @Test
    fun evaluator_staysExactWhereDoubleWouldDrift() {
        // Double 下 1.1 + 2.2 == 3.3000000000000003，精确有理数下必须正好是 33/10
        assertEquals(Fraction.of(33, 10), evaluate("1.1 + 2.2"))
        assertFalse("1.1 + 2.2 不应被当成整数", evaluate("1.1 + 2.2").isInteger)
    }

    // ---------------- 核心：题面求值必须等于答案 ----------------

    @Test
    fun everyGrade_expressionEvaluatesToAnswer() {
        val random = Random(SEED)
        for (grade in Grade.values()) {
            repeat(ROUNDS) {
                for (problem in MathProblemGenerator.generateSet(grade, random = random)) {
                    val value = evaluate(problem.expression)
                    assertTrue(
                        "$grade 的题面求值结果不是整数：${problem.expression} = $value",
                        value.isInteger
                    )
                    assertEquals(
                        "$grade 的题面与答案不一致：${problem.expression} 应为 ${problem.answer}，实际求值 $value",
                        problem.answer.toLong(),
                        value.longValue
                    )
                }
            }
        }
    }

    // ---------------- 不变量 ----------------

    @Test
    fun everyGrade_answerWithinRecognizerRange() {
        val random = Random(SEED)
        for (grade in Grade.values()) {
            repeat(ROUNDS) {
                for (problem in MathProblemGenerator.generateSet(grade, random = random)) {
                    // 答案 0 被排除：画布上单独写一个 "0" 是闭合环，与"什么都没写"难以区分
                    assertTrue(
                        "$grade 的答案必须 ≥ 1：${problem.expression} = ${problem.answer}",
                        problem.answer >= 1
                    )
                    assertTrue(
                        "$grade 的答案超出识别位数上限：${problem.expression} = ${problem.answer}",
                        problem.answer.toString().length <= MathProblemGenerator.ANSWER_DIGITS
                    )
                }
            }
        }
    }

    @Test
    fun everyGrade_expressionsUniqueWithinSet() {
        val random = Random(SEED)
        for (grade in Grade.values()) {
            repeat(ROUNDS) {
                val expressions = MathProblemGenerator.generateSet(grade, random = random)
                    .map { it.expression }
                assertEquals(
                    "$grade 的同一组题里出现重复题面：$expressions",
                    expressions.size,
                    expressions.toSet().size
                )
            }
        }
    }

    /**
     * 减法不能出现负数、除法必须整除 —— 组合运算里两者都可能藏在**中间结果**里
     * （`100 - 6 × 7` 的减法在最后一步，`(320 - 160) ÷ 8` 的除法在最后一步，而
     * `2 + 3 × 4` 的加减在乘除之后），所以这条检查必须跟着求值器一起走完整个优先级栈，
     * 不能像单运算符时代那样"按空格切三段、逐段看运算符"。
     */
    @Test
    fun everyGrade_subtractionNeverNegativeAndDivisionIsExact() {
        val random = Random(SEED)
        for (grade in Grade.values()) {
            repeat(ROUNDS) {
                for (problem in MathProblemGenerator.generateSet(grade, random = random)) {
                    val violations = violationsOf(problem.expression)
                    assertTrue(
                        "$grade 的题面违反不变量：$violations",
                        violations.isEmpty()
                    )
                }
            }
        }
    }

    // ---------------- 年级语义边界 ----------------

    @Test
    fun gradeOne_staysWithin20() {
        val random = Random(SEED)
        repeat(ROUNDS) {
            for (problem in MathProblemGenerator.generateSet(Grade.FIRST, random = random)) {
                assertTrue(
                    "一年级答案必须 ≤ 20：${problem.expression} = ${problem.answer}",
                    problem.answer <= 20
                )
                assertTrue(
                    "一年级不应出现小数或分数：${problem.expression}",
                    "." !in problem.expression && "/" !in problem.expression
                )
            }
        }
    }

    /**
     * 二年级是 100 以内、三年级是三位数以内 —— 这是年级难度分级的产品承诺。
     *
     * 组合运算最容易在这里越界：加减项的取值上限必须跟着乘法/除法的结果一起算，
     * 否则 `81 + 20 = 101` 这种题会静默混进二年级。
     */
    @Test
    fun gradeTwoAndThree_stayWithinTheirNumberRange() {
        val caps = mapOf(Grade.SECOND to 100, Grade.THIRD to 999)
        val random = Random(SEED)
        for ((grade, cap) in caps) {
            repeat(ROUNDS) {
                for (problem in MathProblemGenerator.generateSet(grade, random = random)) {
                    assertTrue(
                        "$grade 的答案超出该年级的数值域（上限 $cap）：${problem.expression} = ${problem.answer}",
                        problem.answer <= cap
                    )
                }
            }
        }
    }

    @Test
    fun gradeFive_expressionsContainDecimals() {
        val random = Random(SEED)
        repeat(ROUNDS) {
            for (problem in MathProblemGenerator.generateSet(Grade.FIFTH, random = random)) {
                assertTrue("五年级题面必须含小数点：${problem.expression}", "." in problem.expression)
            }
        }
    }

    @Test
    fun gradeSix_expressionsAreFractionOrPercent() {
        val random = Random(SEED)
        repeat(ROUNDS) {
            for (problem in MathProblemGenerator.generateSet(Grade.SIXTH, random = random)) {
                assertTrue(
                    "六年级题面应当是分数或百分数：${problem.expression}",
                    "/" in problem.expression || "的" in problem.expression
                )
            }
        }
    }

    /**
     * 四年级应该有相当比例的 4 位答案。
     *
     * 这不是吹毛求疵：三位数乘两位数的积**必然 ≥ 1000**（100×10 就是 1000），而 4 位整串的
     * 实测识别全对率只有 87%。四年级是全功能唯一稳定踩在识别悬崖上的年级，界面上的
     * "两次点击确认"正是为此设计。若哪天有人把乘数范围改小，这条会提醒他重新评估识别率。
     *
     * 新增组合运算的 maker 后本年级的题型从 4 个涨到 6 个，每种题型在一组题里的轮转次数变少，
     * 但 `mulThreeByTwo` 与 `addWithin10000` 仍是每组各出现至少一次，这条断言因此继续成立。
     */
    @Test
    fun gradeFour_producesFourDigitAnswers() {
        val random = Random(SEED)
        repeat(ROUNDS) {
            val problems = MathProblemGenerator.generateSet(Grade.FOURTH, random = random)
            assertTrue(
                "四年级应当稳定出现 4 位答案：${problems.map { it.expression to it.answer }}",
                problems.any { it.answer in 1000..9999 }
            )
        }
    }

    /**
     * 题型覆盖的回归保护：改成"每题独立随机抽题型"后，10 道题全抽到同一种的概率虽小但真实存在。
     *
     * 分类刻意只分**单运算 / 组合运算 / 括号运算**三档，而不是给每个运算符组合点名：组合运算的
     * 具体形态有十来种（`a × b + c`、`a ÷ b - c`、`(a + b) × c` …），逐一点名既维护不住，
     * 也不是产品上真正承诺的东西；而"二年级起有组合运算、三年级起有括号"是明确的承诺。
     *
     * 每个年级都断言**集合相等**（而不是"至少包含"），所以这条同时钉住了两件事：
     * 三年级起每组题必然出现括号题，一年级绝不出现组合运算。
     */
    @Test
    fun everyGrade_coversEveryProblemType() {
        val expected = mapOf(
            Grade.FIRST to setOf(SINGLE),
            Grade.SECOND to setOf(SINGLE, COMBINED),
            Grade.THIRD to setOf(SINGLE, COMBINED, PARENTHESIZED),
            Grade.FOURTH to setOf(SINGLE, COMBINED, PARENTHESIZED),
            Grade.FIFTH to setOf(SINGLE, COMBINED, PARENTHESIZED),
            Grade.SIXTH to setOf(SINGLE, COMBINED, PARENTHESIZED)
        )
        val random = Random(SEED)
        for ((grade, shapes) in expected) {
            repeat(ROUNDS) {
                val actual = MathProblemGenerator.generateSet(grade, random = random)
                    .map { shapeOf(it.expression) }
                    .toSet()
                assertEquals("$grade 的题型没有全部出现：$actual", shapes, actual)
            }
        }
    }

    /**
     * 组合运算必须**真的要求**"先乘除后加减"，否则它只是个样子货。
     *
     * `4 × 5 + 3` 从左到右算与按优先级算同解，学生不知道这条规则也蒙得对；
     * 只有 `3 + 4 × 5` 这种"加减在前"的形态才真的在考它。每个二年级以上的年级都得出现。
     */
    @Test
    fun everyGrade_mixedOperationsRequireOperationOrder() {
        val random = Random(SEED)
        for (grade in Grade.values().filter { it >= Grade.SECOND }) {
            var everSeen = false
            repeat(ROUNDS) {
                if (MathProblemGenerator.generateSet(grade, random = random)
                        .any { requiresOperationOrder(it.expression) }
                ) {
                    everSeen = true
                }
            }
            assertTrue("$grade 的组合运算里没有一条真正考察运算顺序的题面", everSeen)
        }
    }

    /**
     * 小括号的产品承诺：**二年级及以下绝不出现，三年级起必然出现**。
     *
     * 跨 ROUNDS 轮整体判断，而不是逐组判断 —— 括号是"三年级及以上"这个范围的属性，
     * 不是"每一组题都必须有"的硬约束，逐组断言会把生成器的随机性变成测试的脆弱点。
     */
    @Test
    fun parentheses_appearOnlyFromGradeThree() {
        val random = Random(SEED)
        for (grade in Grade.values()) {
            var everSeen = false
            repeat(ROUNDS) {
                if (MathProblemGenerator.generateSet(grade, random = random)
                        .any { "(" in it.expression }
                ) {
                    everSeen = true
                }
            }
            assertEquals(
                "$grade 的小括号使用与约定不符（二年级及以下不应出现，三年级起应当出现）",
                grade >= Grade.THIRD,
                everSeen
            )
        }
    }

    // ---------------- 终止性与确定性 ----------------

    /** timeout 用来抓去重重试的死循环 —— 一年级题面空间最小（约 380 种），最容易在这里挂住。 */
    @Test(timeout = 30_000)
    fun everyGrade_generatesManyRoundsWithoutHanging() {
        val random = Random(SEED)
        for (grade in Grade.values()) {
            repeat(1_000) { MathProblemGenerator.generateSet(grade, random = random) }
        }
    }

    /**
     * 固定种子必须可复现。
     *
     * 这条同时是"实现内部有没有误用 `Random.Default`"的探针 —— 只要有一处 `listOf(...).random()`
     * 忘了传注入的 random，这里就会随机失败。
     */
    @Test
    fun sameSeed_producesIdenticalSet() {
        for (grade in Grade.values()) {
            val first = MathProblemGenerator.generateSet(grade, random = Random(SEED))
            val second = MathProblemGenerator.generateSet(grade, random = Random(SEED))
            assertEquals("$grade 在同一种子下必须可复现", first, second)
        }
    }

    @Test
    fun countIsRespected() {
        assertEquals(1, MathProblemGenerator.generateSet(Grade.THIRD, count = 1).size)
        assertEquals(10, MathProblemGenerator.generateSet(Grade.THIRD).size)
        assertEquals(30, MathProblemGenerator.generateSet(Grade.THIRD, count = 30).size)
    }

    /**
     * 按**运算符个数**与**有无小括号**分类，而不是按运算符的具体组合。
     * 空白符个数 = 运算符个数 + 括号个数，单运算符题面是 2 个空格、组合运算是 4 个。
     */
    private fun shapeOf(expression: String): String = when {
        "(" in expression -> PARENTHESIZED
        expression.count { it == ' ' } >= 4 -> COMBINED
        else -> SINGLE
    }

    /**
     * 题面是否**真的要求**先乘除后加减：加减运算符排在乘除运算符之前、且没有小括号时，
     * 从左到右算与按优先级算会得到不同答案（`3 + 4 × 5` 是 23，从左到右会算成 35）。
     *
     * 括号题不算在内：`(3 + 4) × 5` 考的是括号优先，是另一条规则。
     */
    private fun requiresOperationOrder(expression: String): Boolean {
        if ("(" in expression) return false
        val operators = expression.split(" ").filterIndexed { index, _ -> index % 2 == 1 }
        val firstAdditive = operators.indexOfFirst { it == "+" || it == "-" }
        val firstMultiplicative = operators.indexOfFirst { it == "×" || it == "÷" || it == "的" }
        return firstAdditive >= 0 && firstMultiplicative > firstAdditive
    }

    private companion object {
        const val SEED = 20260913
        const val ROUNDS = 200

        const val SINGLE = "单运算"
        const val COMBINED = "组合运算"
        const val PARENTHESIZED = "括号运算"
    }
}
