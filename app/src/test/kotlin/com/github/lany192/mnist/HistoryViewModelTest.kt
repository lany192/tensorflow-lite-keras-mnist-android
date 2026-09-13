package com.github.lany192.mnist

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** 学习记录页状态机的规格。无参构造、纯 reducer，所以同样不需要 `kotlinx-coroutines-test`。 */
class HistoryViewModelTest {

    private val mistakes = listOf(
        MistakeRecord(Problem("9 - 5", 4), written = "1", mistakeCount = 2),
        MistakeRecord(Problem("2 × 3", 6), written = "1", mistakeCount = 1),
    )

    private fun loaded(intentMistakes: List<MistakeRecord> = mistakes) = HistoryIntent.Loaded(
        stats = listOf(GradeStats(Grade.FIRST, correct = 7, total = 10)),
        sessions = listOf(SessionSummary(1, Grade.FIRST, PracticeSource.GRADE, 42L, 10, 7)),
        mistakes = intentMistakes,
    )

    @Test
    fun init_isEmpty() {
        val vm = HistoryViewModel()

        assertTrue(vm.state.value.sessions.isEmpty())
        assertNull(vm.state.value.overall)
    }

    @Test
    fun loaded_replacesTheWholeState() {
        val vm = HistoryViewModel()

        vm.dispatch(loaded())

        assertEquals(1, vm.state.value.sessions.size)
        assertEquals(2, vm.state.value.mistakes.size)
        assertEquals(70, vm.state.value.overall!!.percent)
    }

    @Test
    fun mistakesBeyondReviewLimit_reportsWhatWasLeftOut() {
        val many = (1..14).map { MistakeRecord(Problem("$it + 1", it + 1), "0", 1) }
        val vm = HistoryViewModel()

        vm.dispatch(loaded(many))

        assertEquals(14 - MathProblemGenerator.DEFAULT_COUNT, vm.state.value.mistakesBeyondReviewLimit)
    }

    @Test
    fun reviewMistakes_emitsStartReviewWithTheMostRecentProblems() {
        val vm = HistoryViewModel()
        vm.dispatch(loaded())

        vm.dispatch(HistoryIntent.ReviewMistakes)

        val effect = runBlocking { vm.effect.first() }
        assertEquals(
            HistoryEffect.StartReview(mistakes.map { it.problem }),
            effect
        )
    }

    /** 错题本为空时不发事件 —— 否则练习页会收到空题目集，表现为"点了没反应"。 */
    @Test
    fun reviewMistakes_withEmptyBook_emitsNothing() = runBlocking {
        val vm = HistoryViewModel()
        vm.dispatch(loaded(emptyList()))

        vm.dispatch(HistoryIntent.ReviewMistakes)

        assertNull(withTimeoutOrNull(50) { vm.effect.first() })
    }
}
