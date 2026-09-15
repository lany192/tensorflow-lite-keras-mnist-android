package com.github.lany192.mnist.history

import com.github.lany192.mnist.data.GradeStats
import com.github.lany192.mnist.data.MistakeRecord
import com.github.lany192.mnist.data.SessionSummary

/**
 * 学习记录页的行模型：整页一个 `RecyclerView`，分区标题、空态、统计行、练习行、错题行都是行型。
 *
 * 这一个文件替换掉的，是原先散在 `HistoryActivity` 三个 `renderXxx()` 里的条件逻辑：
 * 「没有数据就只显示空态」「某个分区为空时连标题都不出现」「错题超出上限才提示」。
 * 那些规则以前只能靠真机肉眼验，现在它们是纯函数，由 `HistoryRowsTest` 钉住。
 *
 * **必须全是 `data class` / `data object`**：`DiffUtil.ItemCallback.areContentsTheSame` 用
 * 结构相等判断"内容变没变"，普通 `class` 会退化成 identity 比较，于是每次渲染都全量重绑。
 *
 * 行里**只放原始数据**（`createdAt` 保持 epoch 毫秒、`percent` 保持整数），不放格式化好的字符串：
 * 文案需要 `Context`，且 `areContentsTheSame` 跑在后台线程，而格式化用的 `SimpleDateFormat`
 * 不是线程安全的。格式化一律在 Adapter 的 `onBindViewHolder` 里做。
 */
sealed interface HistoryRow {

    /** 页面主视觉：累计正确率。仅在有数据时出现。 */
    data class Overall(val percent: Int) : HistoryRow

    /** 空态提示。文案在布局里，不需要绑定。 */
    data object Empty : HistoryRow

    /** 某个年级的累计成绩。 */
    data class GradeStatItem(val stat: GradeStats) : HistoryRow

    /** 「历史练习」分区标题。文案在布局里。 */
    data object SectionSessions : HistoryRow

    /** 一次已归档的练习。 */
    data class SessionItem(val session: SessionSummary) : HistoryRow

    /** 「错题本」分区标题。文案在布局里。 */
    data object SectionMistakes : HistoryRow

    /** 错题数超过一次重做的上限时的提示。 */
    data class MistakesBeyond(val count: Int) : HistoryRow

    /** 错题本的一条。 */
    data class MistakeItem(val mistake: MistakeRecord) : HistoryRow
}

/**
 * 把 [HistoryState] 摊平成行列表。
 *
 * @return 空态恰好是 `[Empty]` 一行；有数据时 `Overall` 在最前，其后按分区依次排列。
 */
fun historyRowsOf(state: HistoryState): List<HistoryRow> {
    val rows = ArrayList<HistoryRow>(state.stats.size + state.sessions.size + state.mistakes.size + 4)

    // 一个判据决定两件事：有没有主视觉、有没有统计行。
    // 原先 textOverall / textEmpty / containerStats 三者的显隐都由 `overall == null` 驱动，
    // 这里必须保持同一个判据 —— 拆成两个条件迟早会漂移成"空态提示和统计行同时出现"。
    val percent = state.overall?.percent
    if (percent == null) {
        rows += HistoryRow.Empty
    } else {
        rows += HistoryRow.Overall(percent)
        state.stats.forEach { rows += HistoryRow.GradeStatItem(it) }
    }

    // 分区标题跟着分区内容走：没有内容时留一个孤零零的标题，看起来像是加载失败。
    if (state.sessions.isNotEmpty()) {
        rows += HistoryRow.SectionSessions
        state.sessions.forEach { rows += HistoryRow.SessionItem(it) }
    }

    if (state.mistakes.isNotEmpty()) {
        rows += HistoryRow.SectionMistakes
        // 提示放在错题条目之前，与原先 titleMistakes → textMistakesBeyond → containerMistakes 的顺序一致
        val beyond = state.mistakesBeyondReviewLimit
        if (beyond > 0) rows += HistoryRow.MistakesBeyond(beyond)
        state.mistakes.forEach { rows += HistoryRow.MistakeItem(it) }
    }

    return rows
}
