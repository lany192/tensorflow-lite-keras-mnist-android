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
 * `Problem("$a + $b", a - b)`）。纯不变量测试（答案非负、≤9999、不重复）对这类错误完全无感，
 * 会全绿放行，所以这段解析代码不能省。
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
 * 题面只有"操作数 运算符 操作数"这一种形状，且操作数内部不含空格，所以按空格切成三段即可
 * 覆盖全部语法（含 `80 的 25%` 这种百分数）。
 */
private fun evaluate(expression: String): Fraction {
    val parts = expression.split(" ")
    assertEquals("题面应当恰好是三段：$expression", 3, parts.size)
    // 百分数必须先于运算符分支判断："的"不在运算符表里，但它同样满足三段形状，只能靠这个分支区分
    if (parts[1] == "的") {
        val percent = parts[2].removeSuffix("%").toLong()
        return parseOperand(parts[0]) * Fraction.of(percent, 100)
    }
    val left = parseOperand(parts[0])
    val right = parseOperand(parts[2])
    return when (parts[1]) {
        "+" -> left + right
        "-" -> left - right
        "×" -> left * right
        "÷" -> left / right
        else -> throw AssertionError("未知运算符：${parts[1]}")
    }
}

private fun parseOperand(token: String): Fraction = when {
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

    /** 求值器必须能判错。一个恒返回期望值的坏求值器会让下面所有生成器测试变成空转。 */
    @Test
    fun evaluator_rejectsWrongExpectedValue() {
        assertNotEquals(Fraction.of(2), evaluate("1/4 + 3/4"))
        assertNotEquals(Fraction.of(5), evaluate("6.3 ÷ 0.9"))
        assertNotEquals(Fraction.of(45), evaluate("23 + 45"))
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

    @Test
    fun everyGrade_subtractionNeverNegativeAndDivisionIsExact() {
        val random = Random(SEED)
        for (grade in Grade.values()) {
            repeat(ROUNDS) {
                for (problem in MathProblemGenerator.generateSet(grade, random = random)) {
                    val parts = problem.expression.split(" ")
                    // 百分数题面的 parts[2] 是 "25%" 而非数字，必须先跳过；它既不是减法也不是除法
                    if (parts[1] == "的") continue
                    val left = parseOperand(parts[0])
                    val right = parseOperand(parts[2])
                    when (parts[1]) {
                        "-" -> assertTrue(
                            "减法不能出现负数：${problem.expression}",
                            (left - right).numerator >= 0
                        )

                        "÷" -> assertTrue(
                            "除法必须整除：${problem.expression} = ${left / right}",
                            (left / right).isInteger
                        )

                        else -> Unit
                    }
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

    /** 题型轮转的回归保护：改成"每题独立随机抽题型"后，10 道题全抽到同一种的概率虽小但真实存在。 */
    @Test
    fun everyGrade_coversEveryProblemType() {
        val expectedTypeCount = mapOf(
            Grade.FIRST to 2,
            Grade.SECOND to 4,
            Grade.THIRD to 4,
            Grade.FOURTH to 4,
            Grade.FIFTH to 4,
            Grade.SIXTH to 5
        )
        val random = Random(SEED)
        for ((grade, typeCount) in expectedTypeCount) {
            repeat(ROUNDS) {
                val types = MathProblemGenerator.generateSet(grade, random = random)
                    .map { operatorOf(it.expression) }
                    .toSet()
                assertEquals("$grade 的题型没有全部出现：$types", typeCount, types.size)
            }
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

    private fun operatorOf(expression: String): String {
        val parts = expression.split(" ")
        return if (parts[1] == "的") "%" else parts[1]
    }

    private companion object {
        const val SEED = 20260913
        const val ROUNDS = 200
    }
}
