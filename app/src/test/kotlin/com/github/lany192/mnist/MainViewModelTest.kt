package com.github.lany192.mnist

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

/**
 * 单数字识别页状态机的可执行规格。
 *
 * **不需要 `kotlinx-coroutines-test`**：ViewModel 内部刻意保持同步（`StateFlow.value` 赋值与
 * `Channel.trySend` 都不挂起），`dispatch` 返回时状态已经就位，直接断言 `state.value` 即可；
 * effect 走无界 Channel，`dispatch` 返回时值已在队列里，`first()` 不会真的挂起。
 */
class MainViewModelTest {

    private val viewModel = MainViewModel(maxDigits = 4)

    private fun takeEffect(): MainEffect = runBlocking { viewModel.effect.first() }

    @Test
    fun init_hasNoResult() {
        assertFalse(viewModel.state.value.hasResult)
        assertEquals("", viewModel.state.value.value)
    }

    @Test
    fun submit_digits_recordsResult() {
        viewModel.dispatch(MainIntent.Submit(DigitInput.Digits(listOf(6, 8), totalCount = 2)))

        assertEquals(listOf(6, 8), viewModel.state.value.digits)
        assertEquals("68", viewModel.state.value.value)
    }

    /** 空画布不改变状态，也不该把上一次的结果擦掉。 */
    @Test
    fun submit_emptyCanvas_keepsPreviousResultAndEmitsEffect() {
        viewModel.dispatch(MainIntent.Submit(DigitInput.Digits(listOf(7), totalCount = 1)))
        viewModel.dispatch(MainIntent.Submit(DigitInput.CanvasEmpty))

        assertEquals(listOf(7), viewModel.state.value.digits)
        assertEquals(MainEffect.EmptyCanvas, takeEffect())
    }

    @Test
    fun submit_notRecognized_keepsPreviousResultAndEmitsEffect() {
        viewModel.dispatch(MainIntent.Submit(DigitInput.NotRecognized))

        assertFalse(viewModel.state.value.hasResult)
        assertEquals(MainEffect.NotRecognized, takeEffect())
    }

    @Test
    fun submit_decimal_recordsPointPosition() {
        viewModel.dispatch(
            MainIntent.Submit(
                DigitInput.Digits(listOf(1, 5), totalCount = 2, decimalIndexes = listOf(1))
            )
        )

        assertEquals(listOf(1, 5), viewModel.state.value.digits)
        assertEquals(listOf(1), viewModel.state.value.decimalIndexes)
        assertEquals("1.5", viewModel.state.value.value)
    }

    @Test
    fun submit_invalidDecimal_keepsPreviousResultAndEmitsEffect() {
        viewModel.dispatch(MainIntent.Submit(DigitInput.Digits(listOf(7), totalCount = 1)))
        viewModel.dispatch(
            MainIntent.Submit(
                DigitInput.Digits(listOf(1, 2), totalCount = 2, decimalIndexes = listOf(0))
            )
        )

        assertEquals("7", viewModel.state.value.value)
        assertEquals(MainEffect.InvalidNumber, takeEffect())
    }

    @Test
    fun submit_tooManyDigits_stillRecordsTruncatedValueAndEmitsEffect() {
        viewModel.dispatch(MainIntent.Submit(DigitInput.Digits(listOf(1, 2, 3, 4), totalCount = 6)))

        assertEquals("1234", viewModel.state.value.value)
        assertEquals(MainEffect.TooManyDigits(4), takeEffect())
    }

    @Test
    fun clear_resetsStateAndEmitsClearCanvas() {
        viewModel.dispatch(MainIntent.Submit(DigitInput.Digits(listOf(1, 2), totalCount = 2)))
        viewModel.dispatch(MainIntent.Clear)

        assertFalse(viewModel.state.value.hasResult)
        assertEquals(MainEffect.ClearCanvas, takeEffect())
    }
}
