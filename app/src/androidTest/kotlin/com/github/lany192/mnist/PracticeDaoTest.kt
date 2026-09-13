package com.github.lany192.mnist

import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 数据库的仪器测试。
 *
 * 只验证**只有设备能验证**的部分：建库、写入、JOIN、字段映射。错题本的规则（取最新一条、
 * 去重、筛出错的）全在纯 Kotlin 的 `HistorySummary` 里，由 `./gradlew test` 覆盖 ——
 * 策略放那边才有 JVM 回归保护，这条边界见 AGENTS.md 的 Persistence 一节。
 */
@RunWith(AndroidJUnit4::class)
class PracticeDaoTest {

    private lateinit var database: PracticeDatabase
    private lateinit var dao: PracticeDao

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        // in-memory：用例之间互不影响，也不会污染设备上真实的练习记录
        database = Room.inMemoryDatabaseBuilder(context, PracticeDatabase::class.java).build()
        dao = database.practiceDao()
    }

    @After
    fun tearDown() {
        database.close()
    }

    private fun session(
        grade: Grade = Grade.FIRST,
        source: PracticeSource = PracticeSource.GRADE,
    ) = PracticeSessionEntity(grade = grade.name, source = source.name, createdAt = 1_000L)

    private fun answer(
        expression: String = "3 + 4",
        correctAnswer: Int = 7,
        written: String = "9",
        correct: Boolean = false,
    ) = AnswerRecordEntity(
        sessionId = 0,
        expression = expression,
        correctAnswer = correctAnswer,
        written = written,
        correct = correct,
    )

    /** session 的 id 必须在同一事务里回填到明细上，否则会留下孤儿行、历史列表也算不出总数。 */
    @Test
    fun insertArchive_writesSessionAndBackfillsItsId() = runBlocking {
        dao.insertArchive(session(), listOf(answer(), answer(expression = "5 + 5", correctAnswer = 10, correct = true)))

        val rows = dao.observeRecentAnswers(100).first()

        assertEquals(2, rows.size)
        assertTrue("明细没有回填 sessionId", rows.all { it.sessionId > 0 })
    }

    /** 重做的记录要写进库（错题本靠它自愈），但不能进累计正确率。 */
    @Test
    fun gradeStats_includesOnlyGradeSessions() = runBlocking {
        dao.insertArchive(session(source = PracticeSource.GRADE), listOf(answer(correct = true)))
        dao.insertArchive(session(source = PracticeSource.REVIEW), listOf(answer(correct = false)))

        val stats = dao.observeGradeStats().first()

        assertEquals(1, stats.size)
        assertEquals(1, stats.single().correct)
        assertEquals(1, stats.single().total)
    }

    @Test
    fun sessions_aggregateTotalsAndKeepBothSources() = runBlocking {
        dao.insertArchive(session(), listOf(answer(correct = true), answer(expression = "1 + 1", correctAnswer = 2)))
        dao.insertArchive(session(source = PracticeSource.REVIEW), listOf(answer(correct = true)))

        val sessions = dao.observeSessions().first()

        assertEquals(2, sessions.size)
        val bySource = sessions.associateBy { it.source }
        assertEquals(2, bySource.getValue(PracticeSource.GRADE.name).total)
        assertEquals(1, bySource.getValue(PracticeSource.REVIEW.name).total)
    }

    /** 「取每个题目的最新一条」依赖这个顺序，所以顺序本身值得钉住。 */
    @Test
    fun recentAnswers_areOrderedByDescendingId() = runBlocking {
        dao.insertArchive(session(), listOf(answer(expression = "first"), answer(expression = "second")))

        val rows = dao.observeRecentAnswers(100).first()

        assertEquals("second", rows.first().expression)
    }

    @Test
    fun recentAnswers_respectsLimit() = runBlocking {
        dao.insertArchive(session(), (1..5).map { answer(expression = "$it + 1") })

        assertEquals(3, dao.observeRecentAnswers(3).first().size)
    }
}
