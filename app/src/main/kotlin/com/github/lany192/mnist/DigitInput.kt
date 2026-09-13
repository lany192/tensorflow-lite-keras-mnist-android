package com.github.lany192.mnist

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
     */
    data class Digits(val value: List<Int>, val totalCount: Int) : DigitInput
}

/**
 * 单次识别允许的最大位数。
 *
 * 实测整串全对率随位数下降（32px 笔宽：2 位 93%、3 位 77%、4 位 87%、5 位 60%），
 * 位数再多每个数字就会被写小，"笔画宽 / 数字高度"越过悬崖，结果必然出错。
 * 与其让用户拿到一个必然是错的结果，不如明确截断并提示。
 *
 * 与 `MathProblemGenerator.ANSWER_DIGITS` 是**同一个物理约束**。这个 4 原本散落在三处、
 * 只靠注释说"它们是同一个约束"，现在由 `DigitInputTest` 断言两者相等。
 */
const val MAX_RECOGNIZED_DIGITS = 4

/** 把画布状态与识别结果归成 [DigitInput]。三种形态互斥且穷尽，不含任何处理策略。 */
fun digitInputOf(canvasEmpty: Boolean, digits: List<Int>, totalCount: Int): DigitInput = when {
    canvasEmpty -> DigitInput.CanvasEmpty
    digits.isEmpty() -> DigitInput.NotRecognized
    else -> DigitInput.Digits(digits, totalCount)
}
