package com.github.lany192.mnist

import androidx.room.Dao
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Transaction
import kotlinx.coroutines.flow.Flow

/**
 * 一次练习。
 *
 * 刻意**不存** `total` / `correct`：那是 [AnswerRecordEntity] 的聚合结果，存两份必然漂移，
 * 历史列表需要时用 `GROUP BY session_id` 现算即可。
 */
@Entity(tableName = "practice_session")
data class PracticeSessionEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    /** 存 `Grade` 的 `name` 而**不是 `ordinal`**：枚举顺序一旦重排，历史数据会静默错位。 */
    val grade: String,
    /** 存 `PracticeSource` 的 `name`。重做（REVIEW）照写不误——错题本靠它自愈——但不计入正确率。 */
    val source: String,
    val createdAt: Long,
)

/**
 * 单题作答明细。错题本从这里查。
 *
 * 不存 `grade`（JOIN session 拿得到）、不存题号（"第 3 题"在几周后的错题本里没有意义）、
 * 不存 `createdAt`（用 session 的时间，避免同一事务内 N 行时间戳不一致导致排序不稳定）。
 */
@Entity(
    tableName = "answer_record",
    foreignKeys = [
        ForeignKey(
            entity = PracticeSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE,
        )
    ],
    indices = [Index("sessionId"), Index("expression")],
)
data class AnswerRecordEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: Long,
    val expression: String,
    val correctAnswer: Int,
    /** 学生写出的数位拼成的串。存 NOT NULL 的空串而不是 NULL，省掉一层 null 语义。 */
    val written: String,
    /**
     * 由 ViewModel 的判定结果**原样写入**，不在 SQL 里用 `CAST(written AS INTEGER)` 重算 ——
     * 那会引入第二套判定实现，而判定必须按数值比较（学生写 "068" 就是 68）。
     */
    val correct: Boolean,
)

/** 按年级聚合的正确率。 */
data class GradeStatsRow(val grade: String, val correct: Int, val total: Int)

/** 一次练习的汇总。`total` / `correct` 由 `answer_record` 现算，不落库。 */
data class SessionRow(
    val id: Long,
    val grade: String,
    val source: String,
    val createdAt: Long,
    val total: Int,
    val correct: Int,
)

@Dao
interface PracticeDao {

    @Insert
    suspend fun insertSession(session: PracticeSessionEntity): Long

    @Insert
    suspend fun insertAnswers(records: List<AnswerRecordEntity>)

    /** 一次练习的 N+1 行必须原子写入，否则会出现"有明细没 session"的孤儿行。 */
    @Transaction
    suspend fun insertArchive(session: PracticeSessionEntity, records: List<AnswerRecordEntity>) {
        val sessionId = insertSession(session)
        insertAnswers(records.map { it.copy(sessionId = sessionId) })
    }

    /**
     * 累计正确率，**只统计按年级出的题**。
     *
     * 重做（`source = 'REVIEW'`）的记录照样写入（错题本要自愈），但不参与统计 ——
     * 刚看过答案的题算进正确率，这个数字就没有意义了。
     */
    @Query(
        """
        SELECT s.grade AS grade, SUM(r.correct) AS correct, COUNT(*) AS total
        FROM answer_record r JOIN practice_session s ON r.sessionId = s.id
        WHERE s.source = 'GRADE'
        GROUP BY s.grade
        """
    )
    fun observeGradeStats(): Flow<List<GradeStatsRow>>

    /** 历史练习列表（含重做）。 */
    @Query(
        """
        SELECT s.id AS id, s.grade AS grade, s.source AS source, s.createdAt AS createdAt,
               COUNT(r.id) AS total, SUM(r.correct) AS correct
        FROM practice_session s LEFT JOIN answer_record r ON r.sessionId = s.id
        GROUP BY s.id
        ORDER BY s.createdAt DESC, s.id DESC
        """
    )
    fun observeSessions(): Flow<List<SessionRow>>

    /**
     * 作答明细的原始行，最近的在前。
     *
     * **刻意不在这里做"只取每个题目的最新一条、再筛出错的"筛选** —— 那段策略放在纯 Kotlin 的
     * `HistorySummary` 里，好让它能在 JVM 单测里被验证（Room 的 in-memory 数据库在本项目没有
     * JVM 路径，见 `AGENTS.md`）。这里只留一条简单到显然正确的查询。
     *
     * 排序用自增 `id` 而不是 `createdAt`：一次练习的 N 行是同一个时间戳，按时间排序不稳定。
     * `LIMIT` 只是内存保护（够上百次练习），不承担任何业务语义。
     */
    @Query("SELECT * FROM answer_record ORDER BY id DESC LIMIT :limit")
    fun observeRecentAnswers(limit: Int): Flow<List<AnswerRecordEntity>>
}
