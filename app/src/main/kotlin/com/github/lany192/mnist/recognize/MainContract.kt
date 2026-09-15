package com.github.lany192.mnist.recognize


/**
 * 单数字识别页的状态。
 *
 * 刻意不含 Bitmap、也不含画布内容：画布属于 View 自己的状态，而 Bitmap 这类 identity-equals
 * 的对象放进 StateFlow 会引起漏渲染或虚假渲染（StateFlow 是按 `equals` 做 conflate 的）。
 */
data class MainState(
    val digits: List<Int> = emptyList(),
    /** 小数点位于第几个数字之前；空表示整数。 */
    val decimalIndexes: List<Int> = emptyList(),
) {
    val hasResult: Boolean get() = digits.isNotEmpty()

    /** 拼成展示用的数值串，如 `[1, 2, 3]` → `"123"`，`[1, 5] + [1]` → `"1.5"`。 */
    val value: String get() = digitsText(digits, decimalIndexes)

    /** 包括小数点在内的字形数量，用于逐字形排查。 */
    val glyphCount: Int get() = digits.size + decimalIndexes.size
}

sealed interface MainIntent {
    /** 点了「识别」。[input] 是 View 采集并识别出的原始结果。 */
    data class Submit(val input: DigitInput) : MainIntent

    /** 点了「清除」。 */
    data object Clear : MainIntent
}

/**
 * 一次性事件。
 *
 * 规则：**凡是新建的 Activity 必须能复现的东西，一律进 [MainState]，不进这里** ——
 * Effect 不跨配置变更存活，放进来的东西旋转后就丢了。
 *
 * 文案由 View 决定，ViewModel 不引用 `R.string`。
 */
sealed interface MainEffect {
    data object ClearCanvas : MainEffect

    /** 画布空白就点了识别。 */
    data object EmptyCanvas : MainEffect

    /** 有笔迹，但切不出任何数字。 */
    data object NotRecognized : MainEffect

    /** 识别到的数字位数超过上限，只取前 [max] 位。 */
    data class TooManyDigits(val max: Int) : MainEffect

    /** 有笔迹但小数点位置非法，例如多个点、点在开头或末尾。 */
    data object InvalidNumber : MainEffect
}
