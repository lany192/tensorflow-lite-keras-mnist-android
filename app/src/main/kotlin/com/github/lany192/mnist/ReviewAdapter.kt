package com.github.lany192.mnist

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.github.lany192.mnist.databinding.ItemReviewLineBinding

/**
 * 结算页「错题回顾」的适配器：一次练习里答错的题，一行一道。
 *
 * 只承载展示 —— 哪几道算错题由 `MathPracticeState.wrongAttempts` 决定（纯 Kotlin，有单测）。
 */
class ReviewAdapter : ListAdapter<Attempt, ReviewAdapter.Holder>(Diff) {

    init {
        // 与 HistoryAdapter 同因、同一条实测结论（含"它当前并不承重"那一段）：
        // 异步填充的适配器照惯例配上，但它不是这里滚动位置的实际保障。
        stateRestorationPolicy = RecyclerView.Adapter.StateRestorationPolicy.PREVENT_WHEN_EMPTY
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder =
        Holder(ItemReviewLineBinding.inflate(LayoutInflater.from(parent.context), parent, false))

    override fun onBindViewHolder(holder: Holder, position: Int) {
        val attempt = getItem(position)
        val context = holder.binding.root.context
        holder.binding.textReviewLine.text = context.getString(
            R.string.math_review_line_format,
            attempt.index,
            attempt.problem.expression,
            attempt.problem.answer,
            // 当前不可达：进 Confirming 的必要条件就是识别出至少一位数字，故 written 必非空。
            attempt.writtenText.ifEmpty { context.getString(R.string.math_review_unrecognized) },
        )
    }

    class Holder(val binding: ItemReviewLineBinding) : RecyclerView.ViewHolder(binding.root)

    private object Diff : DiffUtil.ItemCallback<Attempt>() {
        /** `index` 是一次练习内的题号（1-based），由 `MathPracticeViewModel` 逐题递增，组内唯一。 */
        override fun areItemsTheSame(oldItem: Attempt, newItem: Attempt): Boolean =
            oldItem.index == newItem.index

        override fun areContentsTheSame(oldItem: Attempt, newItem: Attempt): Boolean =
            oldItem == newItem
    }
}
