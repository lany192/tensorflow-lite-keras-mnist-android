package com.github.lany192.mnist.history

import com.github.lany192.mnist.data.AnswerRecordEntity
import com.github.lany192.mnist.data.GradeStats
import com.github.lany192.mnist.data.GradeStatsRow
import com.github.lany192.mnist.data.MistakeRecord
import com.github.lany192.mnist.data.PracticeSource
import com.github.lany192.mnist.data.SessionRow
import com.github.lany192.mnist.data.SessionSummary
import com.github.lany192.mnist.practice.Grade
import com.github.lany192.mnist.practice.MathProblemGenerator
import com.github.lany192.mnist.practice.Problem

/**
 * 学习记录的**全部策略**都在这几个纯函数里，DAO 只负责把原始行取出来。
 *
 * 这样切分的理由是具体的：Room 的 in-memory 数据库在本项目没有 JVM 路径
 * （`room-runtime` 是 KMP 发布，Android 模块的单元测试永远解析到 `-android` 变体），
 * 所以 SQL 里能表达的东西在 `./gradlew test` 里测不到。把"取最新、去重、筛错"搬到纯 Kotlin，
 * 错题本的规则就有了真正的回归保护，仪器测试只需要验证"能建库、能读写"。
 */

/**
 * 错题本的内容。
 *
 * 规则：**每个题目的最新一条记录如果是错的，它就是错题。**
 *
 * 所以"先取最新、再看对错"的顺序不能颠倒 —— 如果先按 `correct = 0` 过滤再取最新，
 * 某题第一次答错、后来答对了，取到的仍然是那条旧的错误记录，这题就**永远留在错题本里**，
 * 重做答对也不会消失。
 *
 * 分组键用 `(expression, correctAnswer)` 而不是只用 `expression`：题干文本会被跨年级复用
 * （一年级的 "3 + 4" 和二年级的 "3 + 4" 是同一串字符），而且只按文本分组等于把
 * "同一串文本 = 同一道题"变成出题器的隐式契约 —— 哪天改出题规则，旧记录会被错误归并。
 *
 * @param records 作答明细，**必须已按自增 id 倒序**（最新的在前）。
 * @param limit 最多返回多少道。
 */
fun mistakesOf(records: List<AnswerRecordEntity>, limit: Int): List<MistakeRecord> {
    val mistakeCounts = records
        .filterNot { it.correct }
        .groupingBy { it.expression to it.correctAnswer }
        .eachCount()
    val handled = HashSet<Pair<String, Int>>()
    val mistakes = ArrayList<MistakeRecord>()
    for (record in records) {
        val key = record.expression to record.correctAnswer
        // 记录已按 id 倒序，所以第一次遇到某个题目时拿到的就是它的最新记录
        if (!handled.add(key)) continue
        if (record.correct) continue
        mistakes += MistakeRecord(
            problem = Problem(record.expression, record.correctAnswer),
            written = record.written,
            mistakeCount = mistakeCounts[key] ?: 1,
        )
        if (mistakes.size >= limit) break
    }
    return mistakes
}

/**
 * 从错题本取一批题去重做。
 *
 * 取最近的 [limit] 道：一次几十道的重做对小学生是惩罚，也会挤掉正常练习。
 * 界面要提示"还有 N 道没纳入"，否则学生会以为数据丢了。
 */
fun reviewProblemsOf(
    mistakes: List<MistakeRecord>,
    limit: Int = MathProblemGenerator.DEFAULT_COUNT,
): List<Problem> = mistakes.take(limit).map { it.problem }

/**
 * 把 DAO 的聚合行转成领域对象。
 *
 * 无法识别的年级名直接跳过，而不是抛异常：数据库里存的是 `Grade.name`，出现陌生值说明数据
 * 被外部改过，那种情况下整页崩掉是最糟的选择。
 *
 * **排序在这里做，因为 DAO 的 `GROUP BY s.grade` 没有 `ORDER BY`**：SQLite 返回的组顺序是任意的
 * （实测近似按 `Grade.name` 的字典序，也就是一年级、五年级、四年级、二年级…）。界面上统计行
 * 按年级从低到高才读得通，而且顺序必须是确定的 —— 否则同一份数据在两次查询里能排出不同的行序，
 * `DiffUtil` 会把它当成一串 move。按 `ordinal` 排同时保证了与年级下拉框的顺序一致。
 */
fun gradeStatsOf(rows: List<GradeStatsRow>): List<GradeStats> = rows
    .mapNotNull { row -> gradeOf(row.grade)?.let { GradeStats(it, row.correct, row.total) } }
    .sortedBy { it.grade.ordinal }

/** 同类转换，理由同上。 */
fun sessionSummariesOf(rows: List<SessionRow>): List<SessionSummary> = rows.mapNotNull { row ->
    val grade = gradeOf(row.grade) ?: return@mapNotNull null
    SessionSummary(
        id = row.id,
        grade = grade,
        source = if (row.source == PracticeSource.REVIEW.name) PracticeSource.REVIEW else PracticeSource.GRADE,
        createdAt = row.createdAt,
        total = row.total,
        correct = row.correct,
    )
}

/** 年级名的解析集中在这里。存的是 `name` 不是 `ordinal`，所以枚举重排不会让历史数据错位。 */
fun gradeOf(name: String): Grade? = Grade.values().firstOrNull { it.name == name }

/**
 * 所有已练习年级的**合计**正确率。
 *
 * 返回 null 表示还没有任何数据 —— 让界面显示"还没有练习记录"，而不是一个会误导人的 "0%"。
 */
fun overallStats(stats: List<GradeStats>): GradeStats? {
    if (stats.isEmpty()) return null
    val correct = stats.sumOf { it.correct }
    val total = stats.sumOf { it.total }
    if (total == 0) return null
    // grade 字段对"合计"没有意义，取第一个只为满足类型；界面上不会用它
    return GradeStats(stats.first().grade, correct, total)
}
