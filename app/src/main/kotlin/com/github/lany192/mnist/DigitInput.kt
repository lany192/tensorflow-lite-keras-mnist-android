package com.github.lany192.mnist

import java.math.BigDecimal

/**
 * 画布采集到的一次输入。
 *
 * 三种形态都是**观察到的事实**，不含任何"该怎么处理"的判断 —— "空画布要不要提示"
 * "切不出数字算不算错""位数超了要不要截断"全部交给 ViewModel 决定。
 *
 * 这样切分事实与策略，是为了让那三条唯一有分支、最容易出 bug 的路径变成可单元测试的
 * 状态转移，而不是散落在 Activity 里、永远测不到的 `if`。
 *
 * 本文件必须保持纯 Kotlin（见 `PureKotlinBoundaryTest`）。
 */
sealed interface DigitInput {
    /** 画布空白，学生什么都没写。 */
    data object CanvasEmpty : DigitInput

    /** 有笔迹，但切不出任何数字（只点了一下、笔画写得太小）。 */
    data object NotRecognized : DigitInput

    /**
     * @param value 识别出的数字位，可能已被 [MAX_RECOGNIZED_DIGITS] 截断。
     * @param totalCount 切分算法实际切出的数字总数，可能大于 [value] 的长度。
     * @param decimalIndexes 小数点位于第几个数字之前。合法数字最多一个小数点，且必须位于
     *   两个数字之间；例如 `value = [0, 2]`、`decimalIndexes = [1]` 表示 `"0.2"`。
     */
    data class Digits(
        val value: List<Int>,
        val totalCount: Int,
        val decimalIndexes: List<Int> = emptyList(),
    ) : DigitInput
}

/**
 * 单次识别允许的最大数字位数。
 *
 * 实测整串全对率随位数下降（32px 笔宽：2 位 93%、3 位 77%、4 位 87%、5 位 60%），
 * 位数再多每个数字就会被写小，"笔画宽 / 数字高度"越过悬崖，结果必然出错。
 * 与其让用户拿到一个必然是错的结果，不如明确截断并提示。
 *
 * 小数点不占这个上限：它是一次合法数字中的额外字形，且最多只能有一个。
 */
const val MAX_RECOGNIZED_DIGITS = 4

/** 把画布状态与识别结果归成 [DigitInput]。三种形态互斥且穷尽，不含任何处理策略。 */
fun digitInputOf(
    canvasEmpty: Boolean,
    digits: List<Int>,
    totalCount: Int,
    decimalIndexes: List<Int> = emptyList(),
): DigitInput = when {
    canvasEmpty -> DigitInput.CanvasEmpty
    digits.isEmpty() -> DigitInput.NotRecognized
    else -> DigitInput.Digits(digits, totalCount, decimalIndexes)
}

/** 将数字位和小数点位置拼回展示文本，保留前导零。 */
fun digitsText(digits: List<Int>, decimalIndexes: List<Int>): String = buildString {
    val points = decimalIndexes.toSet()
    for (index in 0..digits.size) {
        if (index in points) append('.')
        if (index < digits.size) append(digits[index])
    }
}

/**
 * 解析成精确十进制。
 *
 * 合法性比 [digitsText] 更严格：最多一个小数点，且必须夹在数字之间，避免把 `.5`、`5.`
 * 或 `1.2.3` 当成合法答案。返回 [BigDecimal] 而不是 Double，保证 `"68.0"` 与 `68`
 * 能按数值相等比较，也不会引入二进制浮点误差。
 */
fun decimalValueOf(digits: List<Int>, decimalIndexes: List<Int>): BigDecimal? {
    if (digits.isEmpty() || digits.any { it !in 0..9 }) return null
    if (decimalIndexes.size > 1) return null
    val dot = decimalIndexes.singleOrNull()
    if (dot != null && dot !in 1 until digits.size) return null
    return digitsText(digits, decimalIndexes).toBigDecimalOrNull()
}
