package com.github.lany192.mnist

import android.content.res.Resources
import android.view.LayoutInflater
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.github.lany192.mnist.databinding.ItemGradeStatBinding
import com.github.lany192.mnist.databinding.ItemHistoryEmptyBinding
import com.github.lany192.mnist.databinding.ItemHistoryNoteBinding
import com.github.lany192.mnist.databinding.ItemHistoryOverallBinding
import com.github.lany192.mnist.databinding.ItemHistorySectionMistakesBinding
import com.github.lany192.mnist.databinding.ItemHistorySectionSessionsBinding
import com.github.lany192.mnist.databinding.ItemHistorySessionBinding
import com.github.lany192.mnist.databinding.ItemMistakeBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 学习记录页的适配器。整页一个列表，八种行型（见 [HistoryRow]）。
 *
 * 格式化**只在这里**做：行模型里全是原始数据，所以 `DiffUtil` 的后台比较碰不到
 * [timeFormat] 这种非线程安全的格式化器，也不需要 `Context` 参与"内容变没变"的判断。
 */
class HistoryAdapter : ListAdapter<HistoryRow, RecyclerView.ViewHolder>(RowDiff) {

    // 只在主线程使用（onBindViewHolder），所以 SimpleDateFormat 的线程安全问题在这里不存在。
    // 不用 java.time：它需要 API 26+，而本项目 minSdk 24。
    private val timeFormat = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault())

    init {
        // 异步填充的适配器配这个策略是 Android 的惯例：内容还没到时不要丢弃待恢复的滚动位置。
        //
        // **但实测在本项目的两条路径上它都不改变结果**（真机 1080×2340 各测一次）：
        // - 旋屏：有没有它都能停在原处。因为 ViewModel 跨配置变更存活，onCreate 里的
        //   `render(viewModel.state.value)` 在状态恢复之前就已经提交过一次行列表。
        // - 进程被杀后重建：有没有它都回到列表顶部。
        //
        // 保留它是为了 render 的时机一旦变化（比如不再在 onCreate 里同步渲染、或列表改成
        // 分页加载），滚动位置不会无声地丢。**不要据此认为它在当前实现里承重。**
        stateRestorationPolicy = RecyclerView.Adapter.StateRestorationPolicy.PREVENT_WHEN_EMPTY
    }

    override fun getItemViewType(position: Int): Int = when (getItem(position)) {
        is HistoryRow.Overall -> VIEW_TYPE_OVERALL
        is HistoryRow.Empty -> VIEW_TYPE_EMPTY
        is HistoryRow.GradeStatItem -> VIEW_TYPE_GRADE_STAT
        is HistoryRow.SectionSessions -> VIEW_TYPE_SECTION_SESSIONS
        is HistoryRow.SessionItem -> VIEW_TYPE_SESSION
        is HistoryRow.SectionMistakes -> VIEW_TYPE_SECTION_MISTAKES
        is HistoryRow.MistakesBeyond -> VIEW_TYPE_BEYOND
        is HistoryRow.MistakeItem -> VIEW_TYPE_MISTAKE
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return when (viewType) {
            VIEW_TYPE_OVERALL ->
                TextHolder(ItemHistoryOverallBinding.inflate(inflater, parent, false).textOverall)

            VIEW_TYPE_EMPTY ->
                TextHolder(ItemHistoryEmptyBinding.inflate(inflater, parent, false).textEmpty)

            VIEW_TYPE_GRADE_STAT ->
                TextHolder(ItemGradeStatBinding.inflate(inflater, parent, false).textGradeStat)

            VIEW_TYPE_SECTION_SESSIONS -> TextHolder(
                ItemHistorySectionSessionsBinding.inflate(inflater, parent, false).textSectionSessions
            )

            VIEW_TYPE_SESSION ->
                TextHolder(ItemHistorySessionBinding.inflate(inflater, parent, false).textSession)

            VIEW_TYPE_SECTION_MISTAKES -> TextHolder(
                ItemHistorySectionMistakesBinding.inflate(inflater, parent, false).textSectionMistakes
            )

            VIEW_TYPE_BEYOND ->
                TextHolder(ItemHistoryNoteBinding.inflate(inflater, parent, false).textNote)

            VIEW_TYPE_MISTAKE ->
                MistakeHolder(ItemMistakeBinding.inflate(inflater, parent, false))

            else -> error("未知的行型 $viewType")
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val context = holder.itemView.context
        when (val row = getItem(position)) {
            is HistoryRow.Overall -> (holder as TextHolder).text.text =
                context.getString(R.string.history_overall_format, row.percent)

            // 空态与两个分区标题的文案固定在各自的布局里，没有要绑的东西
            is HistoryRow.Empty,
            is HistoryRow.SectionSessions,
            is HistoryRow.SectionMistakes,
            -> Unit

            is HistoryRow.GradeStatItem -> (holder as TextHolder).text.text = context.getString(
                R.string.history_grade_stat_format,
                gradeName(context.resources, row.stat.grade),
                row.stat.correct,
                row.stat.total,
            )

            is HistoryRow.SessionItem -> {
                val grade = gradeName(context.resources, row.session.grade)
                // 重做记录的年级可能跨年级，标出来才不会被误当成按年级练的一组
                val label = if (row.session.source == PracticeSource.REVIEW) {
                    "$grade ${context.getString(R.string.history_session_review_tag)}"
                } else {
                    grade
                }
                (holder as TextHolder).text.text = context.getString(
                    R.string.history_session_format,
                    timeFormat.format(Date(row.session.createdAt)),
                    label,
                    row.session.correct,
                    row.session.total,
                )
            }

            is HistoryRow.MistakesBeyond -> (holder as TextHolder).text.text =
                context.getString(R.string.history_mistakes_beyond_format, row.count)

            is HistoryRow.MistakeItem -> {
                val mistake = (holder as MistakeHolder).binding
                mistake.textMistakeProblem.text = context.getString(
                    R.string.history_mistake_format,
                    row.mistake.problem.expression,
                    row.mistake.problem.answer,
                )
                mistake.textMistakeWritten.text = context.getString(
                    R.string.history_mistake_written_format,
                    // 当前不可达：能落库的作答必然识别出了至少一位数字
                    row.mistake.written.ifEmpty { context.getString(R.string.math_review_unrecognized) },
                    row.mistake.mistakeCount,
                )
            }
        }
    }

    /** 复用 Spinner 那份年级名单。顺序必须与 `Grade` 枚举一致（见 DESIGN.md 9.2）。 */
    private fun gradeName(resources: Resources, grade: Grade): String =
        resources.getStringArray(R.array.grade_names)[grade.ordinal]

    private class TextHolder(val text: TextView) : RecyclerView.ViewHolder(text)

    private class MistakeHolder(val binding: ItemMistakeBinding) :
        RecyclerView.ViewHolder(binding.root)

    private object RowDiff : DiffUtil.ItemCallback<HistoryRow>() {

        /**
         * 身份键。**第一件事永远是先判行型**，而不是让所有行型去比一个统一的 key。
         *
         * DiffUtil 对重复键和撞键**都不会抛异常**：两个"相同"的行却要用不同的布局渲染时，
         * RecyclerView 会拿错误的布局去绑同一个 holder —— 静默错误，只表现为"某一行内容串了"。
         */
        override fun areItemsTheSame(oldItem: HistoryRow, newItem: HistoryRow): Boolean = when {
            oldItem is HistoryRow.Overall && newItem is HistoryRow.Overall -> true

            oldItem is HistoryRow.Empty && newItem is HistoryRow.Empty -> true

            // `GROUP BY s.grade` 保证一个年级至多一行
            oldItem is HistoryRow.GradeStatItem && newItem is HistoryRow.GradeStatItem ->
                oldItem.stat.grade == newItem.stat.grade

            oldItem is HistoryRow.SectionSessions && newItem is HistoryRow.SectionSessions -> true

            // 自增主键
            oldItem is HistoryRow.SessionItem && newItem is HistoryRow.SessionItem ->
                oldItem.session.id == newItem.session.id

            oldItem is HistoryRow.SectionMistakes && newItem is HistoryRow.SectionMistakes -> true

            oldItem is HistoryRow.MistakesBeyond && newItem is HistoryRow.MistakesBeyond -> true

            // 错题的身份是 (expression, correct_answer) 这一对，与 mistakesOf() 的分组键同源。
            // 不能只按 expression 比：题干文本会跨年级复用。
            oldItem is HistoryRow.MistakeItem && newItem is HistoryRow.MistakeItem ->
                oldItem.mistake.problem == newItem.mistake.problem

            else -> false
        }

        override fun areContentsTheSame(oldItem: HistoryRow, newItem: HistoryRow): Boolean =
            oldItem == newItem
    }

    private companion object {
        // 显式常量而不是 hashCode/ordinal：RecyclerView 的回收池按 viewType 分池，
        // 映射一旦不稳定，回收就会串。
        const val VIEW_TYPE_OVERALL = 0
        const val VIEW_TYPE_EMPTY = 1
        const val VIEW_TYPE_GRADE_STAT = 2
        const val VIEW_TYPE_SECTION_SESSIONS = 3
        const val VIEW_TYPE_SESSION = 4
        const val VIEW_TYPE_SECTION_MISTAKES = 5
        const val VIEW_TYPE_BEYOND = 6
        const val VIEW_TYPE_MISTAKE = 7
    }
}
