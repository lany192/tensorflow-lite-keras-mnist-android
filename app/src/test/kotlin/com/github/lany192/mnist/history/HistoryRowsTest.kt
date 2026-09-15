package com.github.lany192.mnist.history

import com.github.lany192.mnist.data.GradeStats
import com.github.lany192.mnist.data.MistakeRecord
import com.github.lany192.mnist.data.PracticeSource
import com.github.lany192.mnist.data.SessionSummary
import com.github.lany192.mnist.practice.Grade
import com.github.lany192.mnist.practice.MathProblemGenerator
import com.github.lany192.mnist.practice.Problem
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 「状态 → 行」的规格。
 *
 * 这几条规则原先散在 `HistoryActivity` 的三个 `renderXxx()` 里，只能靠真机肉眼验：
 * 空态显示什么、某个分区为空时标题还要不要显示、错题超出上限时提示出现在哪。
 * 挪进纯函数之后它们第一次有了回归保护 —— 尤其是"空态"和"分区标题"两条，
 * 写错的后果是界面上多一句无意义的提示，没有任何报错。
 */
class HistoryRowsTest {

    @Test
    fun emptyState_isExactlyOneEmptyRow() {
        assertEquals(listOf(HistoryRow.Empty), historyRowsOf(HistoryState()))
    }

    @Test
    fun withStats_overallComesFirst_withoutEmptyRow() {
        val rows = historyRowsOf(HistoryState(stats = listOf(stat(Grade.THIRD, 8, 10))))

        assertEquals(
            listOf(HistoryRow.Overall(80), HistoryRow.GradeStatItem(stat(Grade.THIRD, 8, 10))),
            rows,
        )
    }

    /**
     * `overall` 为 null 的**两种**来源（没有统计、统计的 total 为 0）必须走同一条分支。
     * 原先 `textOverall` / `textEmpty` / `containerStats` 三者的显隐都由它驱动；
     * 拆成两个条件，迟早会出现"空态提示和统计行同时显示"。
     */
    @Test
    fun zeroTotalStats_showEmptyInsteadOfStatRows() {
        val rows = historyRowsOf(HistoryState(stats = listOf(stat(Grade.THIRD, 0, 0))))

        assertEquals(listOf(HistoryRow.Empty), rows)
    }

    /** 分区标题跟着分区内容走：没有内容时留一个孤零零的标题，看起来像是加载失败。 */
    @Test
    fun emptySections_doNotEmitTheirTitles() {
        val rows = historyRowsOf(HistoryState(stats = listOf(stat(Grade.FIRST, 8, 10))))

        assertFalse(rows.contains(HistoryRow.SectionSessions))
        assertFalse(rows.contains(HistoryRow.SectionMistakes))
    }

    /** 上限之内不提示：多一句"另有 0 道错题"只会让学生困惑。 */
    @Test
    fun mistakesAtTheReviewLimit_haveNoBeyondNote() {
        val rows = historyRowsOf(stateWithMistakes(MathProblemGenerator.DEFAULT_COUNT))

        assertFalse(rows.any { it is HistoryRow.MistakesBeyond })
        assertTrue(rows.contains(HistoryRow.SectionMistakes))
    }

    @Test
    fun mistakesBeyondTheLimit_emitOneNoteRow_beforeTheItems() {
        val rows = historyRowsOf(stateWithMistakes(MathProblemGenerator.DEFAULT_COUNT + 4))

        val noteIndex = rows.indexOf(HistoryRow.MistakesBeyond(4))
        assertTrue("超限提示应当出现", noteIndex >= 0)
        assertTrue(
            "提示要排在错题条目之前，与原先 title → beyond → items 的顺序一致",
            noteIndex < rows.indexOfFirst { it is HistoryRow.MistakeItem },
        )
    }

    /**
     * 没有统计但有练习记录：理论上到不了（完成一组题才归档，归档必然带作答记录），
     * 但行映射得是全函数 —— 没有统计不等于没有练习。
     */
    @Test
    fun sessionsWithoutStats_stillShowTheEmptyRowAndTheirSection() {
        val rows = historyRowsOf(HistoryState(sessions = listOf(session(1L))))

        assertEquals(HistoryRow.Empty, rows.first())
        assertTrue(rows.contains(HistoryRow.SectionSessions))
        assertTrue(rows.contains(HistoryRow.SessionItem(session(1L))))
        assertFalse(rows.any { it is HistoryRow.GradeStatItem })
    }

    /** 行顺序跟着 [HistoryState.stats] 走；它的排序由 `gradeStatsOf` 负责（见 HistorySummaryTest）。 */
    @Test
    fun statRows_followTheOrderOfTheStatsList() {
        val ordered = listOf(stat(Grade.FIRST, 1, 2), stat(Grade.SECOND, 3, 4))
        val rows = historyRowsOf(HistoryState(stats = ordered))

        assertEquals(
            ordered.map { HistoryRow.GradeStatItem(it) },
            rows.filterIsInstance<HistoryRow.GradeStatItem>(),
        )
    }

    private fun stat(grade: Grade, correct: Int, total: Int) = GradeStats(grade, correct, total)

    private fun session(id: Long) = SessionSummary(
        id = id,
        grade = Grade.FIRST,
        source = PracticeSource.GRADE,
        createdAt = 0L,
        total = 10,
        correct = 8,
    )

    private fun stateWithMistakes(count: Int) = HistoryState(
        stats = listOf(stat(Grade.FIRST, 8, 10)),
        mistakes = List(count) { mistake(it) },
    )

    /** 每条的 `(expression, answer)` 都不同 —— 这正是错题本的分组键。 */
    private fun mistake(index: Int) = MistakeRecord(
        problem = Problem("1 + $index", index + 1),
        written = "9",
        mistakeCount = 1,
    )
}
