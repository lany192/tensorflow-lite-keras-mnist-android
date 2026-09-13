package com.github.lany192.mnist

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 错题重做与练习归档的状态机规格。
 *
 * 与 [MathPracticeViewModelTest] 一样不需要 `kotlinx-coroutines-test`：ViewModel 内部保持同步，
 * 归档走的是"reduce 产出、dispatch 交付"的同步路径，没有挂起点。
 */
class MathPracticeViewModelReviewTest {

    private val problems = listOf(Problem("60 + 8", 68), Problem("3 + 4", 7))
    private val mistakes = listOf(Problem("9 - 5", 4), Problem("2 × 3", 6))

    private class RecordingRecorder : PracticeRecorder {
        val archives = mutableListOf<PracticeArchive>()
        override fun record(archive: PracticeArchive) {
            archives += archive
        }
    }

    private fun viewModel(
        list: List<Problem> = problems,
        recorder: PracticeRecorder = PracticeRecorder.NoOp,
    ) = MathPracticeViewModel(generate = { list }, recorder = recorder)

    private fun MathPracticeViewModel.submit(digits: List<Int>) =
        dispatch(MathPracticeIntent.Submit(DigitInput.Digits(digits, digits.size)))

    private fun MathPracticeViewModel.takeEffect(): MathPracticeEffect = runBlocking { effect.first() }

    private fun MathPracticeViewModel.answerWholeSetCorrectly() {
        repeat(state.value.problemCount) {
            submit(state.value.currentProblem.answer.toString().map { it - '0' })
            dispatch(MathPracticeIntent.Confirm)
        }
    }

    // ---------------- 错题重做 ----------------

    @Test
    fun startReview_switchesToReviewModeWithThoseProblems() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.StartReview(mistakes))

        assertTrue(vm.state.value.isReviewing)
        assertEquals(mistakes, vm.state.value.problems)
        assertEquals(0, vm.state.value.index)
        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
    }

    @Test
    fun startReview_emitsClearCanvas() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.StartReview(mistakes))

        assertEquals(MathPracticeEffect.ClearCanvas, vm.takeEffect())
    }

    /**
     * 空列表必须在状态层被挡住。
     *
     * 放行的后果是 `problems[index]` 直接越界崩溃 —— 而空列表是可达的：错题本为空、
     * 或跨页传递的两个数组长度不匹配。
     */
    @Test
    fun startReview_withEmptyList_isIgnored() {
        val vm = viewModel()
        val before = vm.state.value

        vm.dispatch(MathPracticeIntent.StartReview(emptyList()))

        assertEquals(before, vm.state.value)
    }

    /** Activity 在 `onCreate` 里灌注重做题目；旋转会重建 Activity，所以这条必须幂等。 */
    @Test
    fun startReview_whenAlreadyReviewing_isIgnored() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.StartReview(mistakes))
        vm.submit(listOf(9, 9))
        vm.dispatch(MathPracticeIntent.Confirm)
        vm.dispatch(MathPracticeIntent.Next)
        val midway = vm.state.value

        vm.dispatch(MathPracticeIntent.StartReview(mistakes))

        assertEquals(midway, vm.state.value)
    }

    @Test
    fun startReview_resetsProgress() {
        val vm = viewModel()
        vm.answerWholeSetCorrectly()
        assertEquals(MathPracticePhase.Finished, vm.state.value.phase)

        vm.dispatch(MathPracticeIntent.StartReview(mistakes))

        assertEquals(0, vm.state.value.index)
        assertTrue(vm.state.value.attempts.isEmpty())
    }

    @Test
    fun startNewSet_whileReviewing_rerunsSameProblems() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.StartReview(mistakes))
        vm.answerWholeSetCorrectly()

        vm.dispatch(MathPracticeIntent.StartNewSet)

        // 重跑同一批错题，而不是退回按年级出题
        assertTrue(vm.state.value.isReviewing)
        assertEquals(mistakes, vm.state.value.problems)
        assertEquals(0, vm.state.value.index)
        assertTrue(vm.state.value.attempts.isEmpty())
    }

    /**
     * 回归：Spinner 首次布局会带 `position = 0` 回调一次 `SelectGrade`。
     *
     * 如果这次回调被当作"用户在选年级"，重做态就会被自己的初始化回调顶掉 —— 真机上表现为
     * **进了重做页却显示一组全新的年级题**，而且那次练习会被当作普通练习归档、拉低累计正确率。
     */
    @Test
    fun selectGrade_whileReviewing_isIgnored() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.StartReview(mistakes))
        val reviewing = vm.state.value

        vm.dispatch(MathPracticeIntent.SelectGrade(Grade.FIRST))

        assertTrue(vm.state.value.isReviewing)
        assertEquals(reviewing, vm.state.value)
    }

    /** 换个年级也一样忽略：重做态下 Spinner 不可见也不可用，用户操作不到它。 */
    @Test
    fun selectDifferentGrade_whileReviewing_isIgnored() {
        val vm = viewModel()
        vm.dispatch(MathPracticeIntent.StartReview(mistakes))
        val reviewing = vm.state.value

        vm.dispatch(MathPracticeIntent.SelectGrade(Grade.THIRD))

        assertEquals(reviewing, vm.state.value)
    }

    /** 结算页的按钮：只把本次练错的题拿出来重练。 */
    @Test
    fun reviewCurrentMistakes_reviewsOnlyTheWrongOnes() {
        val vm = viewModel()
        vm.submit(problems[0].answer.toString().map { it - '0' })   // 第 1 题：对
        vm.dispatch(MathPracticeIntent.Confirm)
        vm.submit(listOf(9, 9))                                     // 第 2 题：错
        vm.dispatch(MathPracticeIntent.Confirm)
        vm.dispatch(MathPracticeIntent.Next)

        vm.dispatch(MathPracticeIntent.ReviewCurrentMistakes)

        assertTrue(vm.state.value.isReviewing)
        assertEquals(listOf(problems[1]), vm.state.value.problems)
        assertEquals(0, vm.state.value.index)
    }

    @Test
    fun reviewCurrentMistakes_withNoMistakes_isIgnored() {
        val vm = viewModel()
        vm.answerWholeSetCorrectly()
        val finished = vm.state.value

        vm.dispatch(MathPracticeIntent.ReviewCurrentMistakes)

        assertEquals(finished, vm.state.value)
    }

    // ---------------- 归档 ----------------

    @Test
    fun finishGradePractice_archivesWithGradeSource() {
        val recorder = RecordingRecorder()
        val vm = viewModel(recorder = recorder)

        vm.answerWholeSetCorrectly()

        assertEquals(1, recorder.archives.size)
        val archive = recorder.archives.single()
        assertEquals(PracticeSource.GRADE, archive.source)
        assertEquals(Grade.FIRST, archive.grade)
        assertEquals(2, archive.attempts.size)
    }

    /** 重做照样写库（错题本正是靠"最新一条记录"自愈的），但来源标记为 REVIEW，不计入正确率。 */
    @Test
    fun finishReviewPractice_archivesWithReviewSource() {
        val recorder = RecordingRecorder()
        val vm = viewModel(recorder = recorder)
        vm.dispatch(MathPracticeIntent.StartReview(mistakes))

        vm.answerWholeSetCorrectly()

        val archive = recorder.archives.single()
        assertEquals(PracticeSource.REVIEW, archive.source)
        assertEquals(mistakes.size, archive.attempts.size)
    }

    @Test
    fun archive_recordsEveryAttemptNotJustWrongOnes() {
        val recorder = RecordingRecorder()
        val vm = viewModel(recorder = recorder)

        vm.answerWholeSetCorrectly()          // 第 1 题对
        vm.dispatch(MathPracticeIntent.StartNewSet)
        vm.submit(listOf(9, 9))               // 第 1 题错
        vm.dispatch(MathPracticeIntent.Confirm)
        vm.dispatch(MathPracticeIntent.Next)
        vm.submit(problems[1].answer.toString().map { it - '0' })
        vm.dispatch(MathPracticeIntent.Confirm)

        val archive = recorder.archives.last()
        assertEquals(2, archive.attempts.size)
        assertEquals(1, archive.attempts.count { it.correct })
    }

    /** 中途退出不归档。已知取舍：做一半退出会丢掉这半场的记录。 */
    @Test
    fun midSetExit_doesNotArchive() {
        val recorder = RecordingRecorder()
        val vm = viewModel(recorder = recorder)

        vm.submit(problems[0].answer.toString().map { it - '0' })
        vm.dispatch(MathPracticeIntent.Confirm)

        assertEquals(MathPracticePhase.Answering, vm.state.value.phase)
        assertTrue(recorder.archives.isEmpty())
    }

    @Test
    fun archive_carriesTheGradeSelectedForThisSet() {
        val recorder = RecordingRecorder()
        val vm = viewModel(recorder = recorder)
        vm.dispatch(MathPracticeIntent.SelectGrade(Grade.SIXTH))

        vm.answerWholeSetCorrectly()

        assertEquals(Grade.SIXTH, recorder.archives.single().grade)
    }

    /** 默认的 NoOp recorder 必须真的什么都不做，且不能让流程出错。 */
    @Test
    fun defaultRecorder_isNoOpAndDoesNotBreakTheFlow() {
        val vm = viewModel()
        vm.answerWholeSetCorrectly()

        assertEquals(MathPracticePhase.Finished, vm.state.value.phase)
    }
}
