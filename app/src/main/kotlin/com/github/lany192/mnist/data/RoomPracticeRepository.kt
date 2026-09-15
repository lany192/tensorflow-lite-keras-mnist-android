package com.github.lany192.mnist.data

import android.util.Log
import com.github.lany192.mnist.history.gradeStatsOf
import com.github.lany192.mnist.history.mistakesOf
import com.github.lany192.mnist.history.sessionSummariesOf
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

/** 拉到内存里再筛的原始行上限。只是内存保护，不承担业务语义。 */
private const val MISTAKE_SCAN_LIMIT = 1000

/** 错题本最多展示多少条。 */
private const val MISTAKE_LIMIT = 100

private const val TAG = "PracticeRepo"

/**
 * [PracticeRepository] / [PracticeRecorder] 的 Room 实现。
 *
 * 换了存储方案（KSP 走不通、或改用别的库）只需要换掉这个文件 —— 接口、ViewModel、测试、
 * 界面全都不动。
 */
class RoomPracticeRepository(
    private val dao: PracticeDao,
    private val now: () -> Long = System::currentTimeMillis,
) : PracticeRepository, PracticeRecorder {

    /**
     * 应用级作用域，**刻意不挂任何 Activity**。
     *
     * 这正是 [PracticeRecorder] 不让 View 层写库的原因：结算瞬间旋转屏幕会取消 Activity 的
     * `lifecycleScope`，写入夭折，而触发它的 Effect 早已被消费、不会重发 —— 结果是结算页
     * 正常显示、数据一条没写、没有任何报错。挂在这里，Activity 的死活与归档无关。
     */
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun record(archive: PracticeArchive) {
        if (archive.attempts.isEmpty()) return
        scope.launch {
            // 归档失败不能影响学生的结算页（那时已经渲染完了），所以吞掉异常只留日志。
            // 刻意不弹 Toast —— 那会把一次后台故障变成打断学生的弹窗。
            runCatching { dao.insertArchive(archive.toSessionEntity(now()), archive.toAnswerEntities()) }
                .onFailure { Log.w(TAG, "练习归档失败", it) }
        }
    }

    // 读取走「取原始行 → 纯 Kotlin 策略」：策略放在 HistorySummary 里才有 JVM 单测覆盖，
    // Room 的 in-memory 数据库在本项目没有 JVM 路径（见 AGENTS.md）。

    override fun observeGradeStats(): Flow<List<GradeStats>> =
        dao.observeGradeStats().map(::gradeStatsOf)

    override fun observeSessions(): Flow<List<SessionSummary>> =
        dao.observeSessions().map(::sessionSummariesOf)

    override fun observeMistakes(): Flow<List<MistakeRecord>> =
        dao.observeRecentAnswers(MISTAKE_SCAN_LIMIT).map { mistakesOf(it, MISTAKE_LIMIT) }
}

private fun PracticeArchive.toSessionEntity(createdAt: Long) = PracticeSessionEntity(
    // 存枚举的 name 不存 ordinal：枚举顺序一旦重排，历史数据会静默错位
    grade = grade.name,
    source = source.name,
    createdAt = createdAt,
)

private fun PracticeArchive.toAnswerEntities(): List<AnswerRecordEntity> = attempts.map { attempt ->
    AnswerRecordEntity(
        // 真实的 session id 由 insertArchive 在插入 session 后回填
        sessionId = 0,
        expression = attempt.problem.expression,
        correctAnswer = attempt.problem.answer,
        written = attempt.writtenText,
        // 判定结果原样搬运，不在 SQL 里重算 —— 判定必须按数值比较（"068" 就是 68）
        correct = attempt.correct,
    )
}
