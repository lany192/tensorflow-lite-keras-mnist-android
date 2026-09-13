package com.github.lany192.mnist

import org.junit.Assert.assertEquals
import org.junit.Test

/** [digitInputOf] 的纯 JVM 测试：三种形态互斥，且每种形态的判定优先级是明确的。 */
class DigitInputTest {

    @Test
    fun emptyCanvas_isCanvasEmpty() {
        assertEquals(DigitInput.CanvasEmpty, digitInputOf(canvasEmpty = true, digits = emptyList(), totalCount = 0))
    }

    /** 画布空优先于其他判断：即便识别结果非空（不该发生），也应报"什么都没写"。 */
    @Test
    fun emptyCanvas_winsOverDigits() {
        assertEquals(DigitInput.CanvasEmpty, digitInputOf(canvasEmpty = true, digits = listOf(1, 2), totalCount = 2))
    }

    @Test
    fun inkButNoDigits_isNotRecognized() {
        assertEquals(DigitInput.NotRecognized, digitInputOf(canvasEmpty = false, digits = emptyList(), totalCount = 0))
    }

    /** 超位数时 value 已被截断、totalCount 仍如实反映实际切出的位数，两者一起传递不丢信息。 */
    @Test
    fun digits_carryValueAndTotalCount() {
        assertEquals(
            DigitInput.Digits(listOf(6, 8), totalCount = 7),
            digitInputOf(canvasEmpty = false, digits = listOf(6, 8), totalCount = 7)
        )
    }

    /**
     * 这个 4 同时约束着识别截断与出题答案的位数上限，两处一旦漂移就会出现
     * "题目答案 4 位、识别只认 3 位"这种静默错误。
     */
    @Test
    fun maxRecognizedDigits_matchesGeneratorAnswerDigits() {
        assertEquals(MathProblemGenerator.ANSWER_DIGITS, MAX_RECOGNIZED_DIGITS)
    }
}
