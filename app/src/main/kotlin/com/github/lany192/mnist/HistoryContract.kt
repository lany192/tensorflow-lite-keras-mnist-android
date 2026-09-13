package com.github.lany192.mnist

/**
 * 学习记录页的状态。
 *
 * 全部由 Activity 从 Room 的 Flow 收集后灌进来 —— **读可以重推**（Flow 每次订阅都会重发），
 * 所以这个 ViewModel 不需要持有数据库、不需要协程、也不需要工厂。
 */
data class HistoryState(
    val stats: List<GradeStats> = emptyList(),
    val sessions: List<SessionSummary> = emptyList(),
    val mistakes: List<MistakeRecord> = emptyList(),
) {
    /** 所有年级合计的正确率；没有任何记录时为 null。 */
    val overall: GradeStats? get() = overallStats(stats)

    /** 错题本超出一次重做上限的条数。界面要提示它，否则学生会以为错题丢了。 */
    val mistakesBeyondReviewLimit: Int
        get() = (mistakes.size - MathProblemGenerator.DEFAULT_COUNT).coerceAtLeast(0)
}

sealed interface HistoryIntent {
    /** 数据库有变化时由 View 灌入。 */
    data class Loaded(
        val stats: List<GradeStats>,
        val sessions: List<SessionSummary>,
        val mistakes: List<MistakeRecord>,
    ) : HistoryIntent

    /** 拿错题本里的题去重做。 */
    data object ReviewMistakes : HistoryIntent
}

/** 一次性事件。 */
sealed interface HistoryEffect {
    /** 带着题目跳到练习页。 */
    data class StartReview(val problems: List<Problem>) : HistoryEffect
}
