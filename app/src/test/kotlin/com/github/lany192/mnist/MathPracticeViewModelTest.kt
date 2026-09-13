package com.github.lany192.mnist

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 练习页状态机的可执行规格。
 *
 * **不需要 `kotlinx-coroutines-test`**：ViewModel 内部刻意保持同步（见其类注释），
 * `dispatch` 返回时状态已经就位，直接断言 `state.value`；effect 走无界 Channel，
 * 值在 `dispatch` 返回时已在队列里，`first()` 不会真的挂起。
 */
class MathPracticeViewModelTest {

    /** 注入固定题目，让每条断言都完全确定 —— 这正是 `generate` 参数存在的原因。 */
    private val problems = listOf(
        Problem("60 + 8", 68),
        Problem("3 + 4", 7),
        Problem("9 - 5", 4),
    )

    private fun viewModel(
        list: List<Problem> = problems,
        maxDigits: Int = 4,
    ) = MathPracticeViewModel(generate = { list }, maxDigits = maxDigits)

    private fun MathPracticeViewModel.submit(digits: List<Int>, totalCount: Int = digits.size) =
        dispatch(MathPracticeIntent.Submit(DigitInput.Digits(digits, totalCount)))

    private fun MathPracticeViewModel.takeEffect(): MathPracticeEffect = runBlocking { effect.first() }

    /** 走完「写出正确答案 → 确认」的完整路径。 */
    private fun MathPracticeViewModel.answerCorrectly() {
        submit(state.value.currentProblem.answer.toString().map { it - '0' })
        dispatch(MathPracticeIntent.Confirm)
    }

    // ---------------- 初始化 ----------------

    @Test
    fun init_startsAnsweringAtFirstProblem() {
        val vm = viewModel()

        assertEquals(Grade.FIRST, vm.state.value.grade)
        assertEquals(0, vm.state.value.index)
        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertTrue(vm.state.value.attempts.isEmpty())
    }

    @Test
    fun init_problemsComeFromInjectedGenerator() {
        assertEquals(problems, viewModel().state.value.problems)
    }

    // ---------------- 提交的三条策略路径（DigitInput 换来的可测性）----------------

    @Test
    fun submit_emptyCanvas_staysAnsweringAndEmitsEmptyCanvas() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.Submit(DigitInput.CanvasEmpty))

        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertEquals(MathPracticeEffect.EmptyCanvas, vm.takeEffect())
    }

    @Test
    fun submit_nothingRecognized_staysAnsweringAndEmitsNotRecognized() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.Submit(DigitInput.NotRecognized))

        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertEquals(MathPracticeEffect.NotRecognized, vm.takeEffect())
    }

    @Test
    fun submit_digits_entersConfirmingWithThoseDigits() {
        val vm = viewModel()
        vm.submit(listOf(6, 8))

        assertEquals(MathPracticePhase.Confirming(listOf(6, 8)), vm.state.value.phase)
    }

    @Test
    fun submit_tooManyDigits_truncatesButStillEntersConfirming() {
        val vm = viewModel(maxDigits = 2)
        vm.submit(listOf(1, 2), totalCount = 5)

        assertEquals(MathPracticePhase.Confirming(listOf(1, 2)), vm.state.value.phase)
        assertEquals(MathPracticeEffect.TooManyDigits(2), vm.takeEffect())
    }

    // ---------------- 判定 ----------------

    @Test
    fun confirm_correct_recordsAttemptAndAdvancesImmediately() {
        val vm = viewModel()
        vm.answerCorrectly()

        // 答对不停顿：判定完就已经不在这道题上了
        assertEquals(1, vm.state.value.index)
        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertEquals(listOf(Attempt(1, problems[0], listOf(6, 8), true)), vm.state.value.attempts)
    }

    /** 数字比较而非字符串比较：学生写 "068"，与答案 68 是同一个数。 */
    @Test
    fun confirm_withLeadingZero_isStillCorrect() {
        val vm = viewModel()
        vm.submit(listOf(0, 6, 8))
        vm.dispatch(MathPracticeIntent.Confirm)

        assertTrue(vm.state.value.attempts.single().correct)
    }

    @Test
    fun submit_decimal_entersConfirmingWithPointPosition() {
        val vm = viewModel()
        vm.dispatch(
            MathPracticeIntent.Submit(
                DigitInput.Digits(listOf(6, 8, 0), totalCount = 3, decimalIndexes = listOf(2))
            )
        )

        assertEquals(MathPracticePhase.Confirming(listOf(6, 8, 0), listOf(2)), vm.state.value.phase)
    }

    @Test
    fun submit_invalidDecimal_staysAnsweringAndEmitsInvalidNumber() {
        val vm = viewModel()
        vm.dispatch(
            MathPracticeIntent.Submit(
                DigitInput.Digits(listOf(1, 2, 3), totalCount = 3, decimalIndexes = listOf(1, 2))
            )
        )

        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertEquals(MathPracticeEffect.InvalidNumber, vm.takeEffect())
    }

    /** 精确数值比较：答案 68 时，学生写 "68.0" 也应当判对。 */
    @Test
    fun confirm_decimalFormOfInteger_isCorrect() {
        val vm = viewModel()
        vm.dispatch(
            MathPracticeIntent.Submit(
                DigitInput.Digits(listOf(6, 8, 0), totalCount = 3, decimalIndexes = listOf(2))
            )
        )
        vm.dispatch(MathPracticeIntent.Confirm)

        assertEquals(1, vm.state.value.index)
        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertEquals("68.0", vm.state.value.attempts.single().writtenText)
        assertTrue(vm.state.value.attempts.single().correct)
    }

    @Test
    fun confirm_wrong_entersJudgedWithoutAdvancing() {
        val vm = viewModel()
        vm.submit(listOf(9, 9))
        vm.dispatch(MathPracticeIntent.Confirm)

        assertEquals(MathPracticePhase.Judged(listOf(9, 9)), vm.state.value.phase)
        assertEquals(0, vm.state.value.index)
        assertFalse(vm.state.value.attempts.single().correct)
    }

    @Test
    fun next_afterJudged_movesOnAndEmitsClearCanvas() {
        val vm = viewModel()
        vm.submit(listOf(9, 9))
        vm.dispatch(MathPracticeIntent.Confirm)
        vm.dispatch(MathPracticeIntent.Next)

        assertEquals(1, vm.state.value.index)
        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertEquals(MathPracticeEffect.ClearCanvas, vm.takeEffect())
    }

    @Test
    fun confirm_onLastProblemWrong_finishesOnlyAfterNext() {
        val vm = viewModel(list = listOf(problems[0]))
        vm.submit(listOf(9, 9))
        vm.dispatch(MathPracticeIntent.Confirm)
        assertEquals(MathPracticePhase.Judged(listOf(9, 9)), vm.state.value.phase)

        vm.dispatch(MathPracticeIntent.Next)
        assertEquals(MathPracticePhase.Finished, vm.state.value.phase)
    }

    @Test
    fun confirm_onLastProblemCorrect_finishesImmediately() {
        val vm = viewModel(list = listOf(problems[0]))
        vm.answerCorrectly()

        assertEquals(MathPracticePhase.Finished, vm.state.value.phase)
    }

    // ---------------- 重复点击的第二道防线 ----------------

    @Test
    fun confirm_whenNotConfirming_isIgnored() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.Confirm)

        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertTrue(vm.state.value.attempts.isEmpty())
    }

    @Test
    fun next_whenNotJudged_isIgnored() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.Next)

        assertEquals(0, vm.state.value.index)
        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
    }

    // ---------------- 重写 / 清除 / 换组 ----------------

    @Test
    fun rewrite_fromConfirming_returnsToAnsweringAndEmitsClearCanvas() {
        val vm = viewModel()
        vm.submit(listOf(6, 8))
        vm.dispatch(MathPracticeIntent.Rewrite)

        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertEquals(MathPracticeEffect.ClearCanvas, vm.takeEffect())
    }

    @Test
    fun clearCanvas_keepsPhase() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.ClearCanvas)

        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertEquals(MathPracticeEffect.ClearCanvas, vm.takeEffect())
    }

    @Test
    fun startNewSet_resetsIndexAndAttempts_keepsGrade() {
        val vm = viewModel()
        vm.answerCorrectly()
        vm.dispatch(MathPracticeIntent.StartNewSet)

        assertEquals(0, vm.state.value.index)
        assertTrue(vm.state.value.attempts.isEmpty())
        assertEquals(Grade.FIRST, vm.state.value.grade)
    }

    @Test
    fun selectGrade_different_regeneratesAndResets() {
        val vm = viewModel()
        vm.answerCorrectly()
        vm.dispatch(MathPracticeIntent.SelectGrade(Grade.FOURTH))

        assertEquals(Grade.FOURTH, vm.state.value.grade)
        assertEquals(0, vm.state.value.index)
        assertTrue(vm.state.value.attempts.isEmpty())
        assertEquals(problems, vm.state.value.problems)
    }

    /** Spinner 首次布局会用 position=0 回调一次；这条闸门保证它不会白出一轮题、不会清空进度。 */
    @Test
    fun selectGrade_same_isNoOp() {
        val vm = viewModel()
        vm.answerCorrectly()
        val before = vm.state.value

        vm.dispatch(MathPracticeIntent.SelectGrade(Grade.FIRST))

        assertEquals(before, vm.state.value)
    }

    // ---------------- 边界 ----------------

    /** 结算后 index 停在最后一题上，`problems[index]` 永远合法（旧实现会越界，只是侥幸没被索引到）。 */
    @Test
    fun finished_stopsIndexAtLastProblem() {
        val vm = viewModel()
        repeat(problems.size) { vm.answerCorrectly() }

        assertEquals(MathPracticePhase.Finished, vm.state.value.phase)
        assertEquals(problems.lastIndex, vm.state.value.index)
        assertEquals(problems.last(), vm.state.value.currentProblem)
    }

    @Test
    fun attemptsAndCounts_accumulateAcrossWholeSet() {
        val vm = viewModel()
        vm.answerCorrectly()                 // 第 1 题：对
        vm.submit(listOf(9, 9))              // 第 2 题：错
        vm.dispatch(MathPracticeIntent.Confirm)
        vm.dispatch(MathPracticeIntent.Next)
        vm.answerCorrectly()                 // 第 3 题：对，且是最后一题

        assertEquals(3, vm.state.value.attempts.size)
        assertEquals(2, vm.state.value.correctCount)
        assertEquals(1, vm.state.value.wrongAttempts.size)
        assertEquals(MathPracticePhase.Finished, vm.state.value.phase)
    }
}
