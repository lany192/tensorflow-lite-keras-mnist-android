package com.github.lany192.mnist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 错题本与统计的纯函数规格。
 *
 * 这些用例是本次改动里最值钱的一批：`mistakesOf` 的规则只有这一处实现，而它随手写错的后果
 * 是"错题本永远不消失"这种**没有报错**的功能失效。
 */
class HistorySummaryTest {

    /** DAO 返回的行是**按 id 倒序**的，最新的在前 —— 所有用例都按这个前提构造。 */
    private fun record(
        id: Long,
        expression: String = "3 + 4",
        answer: Int = 7,
        written: String = "9",
        correct: Boolean = false,
    ) = AnswerRecordEntity(
        id = id,
        sessionId = 1,
        expression = expression,
        correctAnswer = answer,
        written = written,
        correct = correct,
    )

    // ---------------- 错题本 ----------------

    /**
     * 核心回归：某题先答错、后来答对，它必须从错题本消失。
     *
     * 如果实现写成"先按 `correct = 0` 过滤、再取最新"，取到的会是那条旧的错误记录，
     * 这题就永远留在错题本里 —— 学生订正了也看不到任何变化。
     */
    @Test
    fun mistakes_problemCorrectedLater_disappears() {
        val records = listOf(
            record(id = 2, correct = true),
            record(id = 1, correct = false),
        )

        assertTrue(mistakesOf(records, limit = 10).isEmpty())
    }

    @Test
    fun mistakes_problemFailedLastTime_isIncluded() {
        val records = listOf(
            record(id = 2, correct = false, written = "9"),
            record(id = 1, correct = true),
        )

        val mistakes = mistakesOf(records, limit = 10)
        assertEquals(1, mistakes.size)
        assertEquals(Problem("3 + 4", 7), mistakes.single().problem)
        assertEquals("9", mistakes.single().written)
    }

    @Test
    fun mistakes_sameProblemFailedRepeatedly_appearsOnce() {
        val records = listOf(
            record(id = 3, correct = false),
            record(id = 2, correct = false),
        )

        assertEquals(1, mistakesOf(records, limit = 10).size)
    }

    @Test
    fun mistakes_countEveryWrongAttempt() {
        val records = listOf(
            record(id = 3, correct = false),
            record(id = 2, correct = false),
            record(id = 1, correct = true),
        )

        assertEquals(2, mistakesOf(records, limit = 10).single().mistakeCount)
    }

    /**
     * 分组键必须带上正确答案。
     *
     * 题干文本会被跨年级复用（一年级的 "3 + 4" 和二年级的 "3 + 4" 是同一串字符），
     * 只按文本分组会把两道不同的题错误归并成一道。
     */
    @Test
    fun mistakes_sameExpressionDifferentAnswer_areDifferentProblems() {
        val records = listOf(
            record(id = 2, expression = "3 + 4", answer = 7, correct = false),
            record(id = 1, expression = "3 + 4", answer = 8, correct = false),
        )

        assertEquals(2, mistakesOf(records, limit = 10).size)
    }

    @Test
    fun mistakes_respectsLimit_keepingTheMostRecent() {
        val records = (5 downTo 1).map {
            record(id = it.toLong(), expression = "$it + 1", answer = it + 1, correct = false)
        }

        val mistakes = mistakesOf(records, limit = 2)

        assertEquals(listOf("5 + 1", "4 + 1"), mistakes.map { it.problem.expression })
    }

    @Test
    fun mistakes_allCorrect_yieldsEmpty() {
        val records = listOf(record(id = 2, correct = true), record(id = 1, correct = true))

        assertTrue(mistakesOf(records, limit = 10).isEmpty())
    }

    // ---------------- 正确率 ----------------

    @Test
    fun overallStats_sumsAcrossGrades() {
        val stats = listOf(
            GradeStats(Grade.FIRST, correct = 8, total = 10),
            GradeStats(Grade.SECOND, correct = 6, total = 10),
        )

        val overall = overallStats(stats)!!
        assertEquals(14, overall.correct)
        assertEquals(20, overall.total)
        assertEquals(70, overall.percent)
    }

    @Test
    fun overallStats_noData_returnsNullSoUiCanSaySoInsteadOfZeroPercent() {
        assertNull(overallStats(emptyList()))
    }

    @Test
    fun gradeStats_percentIsNullWhenNothingRecorded() {
        assertNull(GradeStats(Grade.FIRST, correct = 0, total = 0).percent)
    }

    /** 数据库里存的是枚举的 `name`；出现陌生值说明数据被外部改过，跳过即可，不该整页崩掉。 */
    @Test
    fun gradeStatsOf_skipsUnknownGradeName() {
        val rows = listOf(
            GradeStatsRow(grade = "FIRST", correct = 1, total = 2),
            GradeStatsRow(grade = "NOT_A_GRADE", correct = 1, total = 1),
        )

        val stats = gradeStatsOf(rows)

        assertEquals(1, stats.size)
        assertEquals(Grade.FIRST, stats.single().grade)
    }

    /**
     * 统计按年级从低到高排。
     *
     * 排序放在这里而不是界面上，是因为 DAO 的 `GROUP BY s.grade` **没有 `ORDER BY`**：
     * SQLite 返回的组顺序是任意的（实测近似按枚举名的字典序，也就是一、五、四、二、六、三年级）。
     * 界面按这个顺序渲染，"统计"这一栏就没法读了；而且顺序不确定意味着同一份数据两次查询能排
     * 出不同的行序，列表的差异计算会把它们当成一串 move。
     */
    @Test
    fun gradeStatsOf_sortsByGradeOrder_regardlessOfInputOrder() {
        val rows = listOf(
            GradeStatsRow(grade = "FOURTH", correct = 1, total = 2),
            GradeStatsRow(grade = "FIRST", correct = 1, total = 2),
            GradeStatsRow(grade = "SECOND", correct = 1, total = 2),
        )

        val stats = gradeStatsOf(rows)

        assertEquals(listOf(Grade.FIRST, Grade.SECOND, Grade.FOURTH), stats.map { it.grade })
    }

    @Test
    fun sessionSummariesOf_mapsSourceAndGrade() {
        val rows = listOf(
            SessionRow(id = 1, grade = "SIXTH", source = "REVIEW", createdAt = 42L, total = 3, correct = 1),
        )

        val summary = sessionSummariesOf(rows).single()

        assertEquals(Grade.SIXTH, summary.grade)
        assertEquals(PracticeSource.REVIEW, summary.source)
        assertEquals(42L, summary.createdAt)
    }

    // ---------------- 重做取题 ----------------

    @Test
    fun reviewProblemsOf_takesAtMostOneSetWorth() {
        val mistakes = (1..15).map { MistakeRecord(Problem("$it + 1", it + 1), written = "0", mistakeCount = 1) }

        assertEquals(MathProblemGenerator.DEFAULT_COUNT, reviewProblemsOf(mistakes).size)
        assertEquals(3, reviewProblemsOf(mistakes, limit = 3).size)
    }
}
