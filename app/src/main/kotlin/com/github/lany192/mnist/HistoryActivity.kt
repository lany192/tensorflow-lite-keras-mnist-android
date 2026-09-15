package com.github.lany192.mnist

import android.os.Bundle
import android.view.View
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.github.lany192.mnist.databinding.ActivityHistoryBinding
import com.github.lany192.mnist.databinding.ItemGradeStatBinding
import com.github.lany192.mnist.databinding.ItemHistorySessionBinding
import com.github.lany192.mnist.databinding.ItemMistakeBinding
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 学习记录页。MVI：View 订阅 Room 的 Flow 后灌进 [HistoryViewModel]，再按状态渲染界面。
 *
 * 数据流方向与练习页**相反**是刻意的 —— 判据是"这次操作的结果可不可以重新推导"：
 * **读可以重推**（Flow 每次订阅都会重发），所以读取由 View 层收集即可，这个 ViewModel
 * 保持无参构造、纯 reducer；而**写不可重推**，所以练习页必须把归档端口注入进 ViewModel。
 */
class HistoryActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryBinding
    private val viewModel: HistoryViewModel by viewModels()

    // 只在主线程使用，所以 SimpleDateFormat 的线程安全问题在这里不存在。
    // 不用 java.time：它需要 API 26+，而本项目 minSdk 24。
    private val timeFormat = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val repository = PracticeDatabase.repository(applicationContext)

        binding.buttonBack.setOnClickListener { finish() }
        binding.buttonReviewMistakes.setOnClickListener {
            viewModel.dispatch(HistoryIntent.ReviewMistakes)
        }

        render(viewModel.state.value)
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { viewModel.state.collect(::render) }
                launch { viewModel.effect.collect(::handleEffect) }
                // 三个 Flow 合流后再灌：分开收集会让界面出现"统计已更新、错题还是旧的"这种中间态
                launch {
                    combine(
                        repository.observeGradeStats(),
                        repository.observeSessions(),
                        repository.observeMistakes(),
                    ) { stats, sessions, mistakes ->
                        HistoryIntent.Loaded(stats, sessions, mistakes)
                    }.collect(viewModel::dispatch)
                }
            }
        }
    }

    private fun handleEffect(effect: HistoryEffect) {
        when (effect) {
            is HistoryEffect.StartReview ->
                startActivity(MathPracticeActivity.reviewIntent(this, effect.problems))
        }
    }

    /** 幂等。这个页面没有画布，所以可以直接用 GONE/VISIBLE 而不必担心挤动别的控件。 */
    private fun render(state: HistoryState) {
        val percent = state.overall?.percent
        binding.textOverall.text = percent?.let { getString(R.string.history_overall_format, it) } ?: ""

        val empty = state.overall == null
        binding.textEmpty.visibility = if (empty) View.VISIBLE else View.GONE
        binding.containerStats.visibility = if (empty) View.GONE else View.VISIBLE

        // 主视觉跟着内容走。空态下"累计正确率 X%"没有值可写（上面写的是空串），
        // 留着它会在空提示上方压出一整块留白；有数据时它才是这一页的主角。
        binding.textOverall.visibility = if (empty) View.GONE else View.VISIBLE

        renderStats(state)
        renderSessions(state)
        renderMistakes(state)
    }

    private fun renderStats(state: HistoryState) {
        val container = binding.containerStats
        container.removeAllViews()
        for (stat in state.stats) {
            val item = ItemGradeStatBinding.inflate(layoutInflater, container, false)
            item.textGradeStat.text = getString(
                R.string.history_grade_stat_format,
                gradeName(stat.grade), stat.correct, stat.total,
            )
            container.addView(item.root)
        }
    }

    private fun renderSessions(state: HistoryState) {
        // 分区标题跟着分区内容走：没有内容时留一个孤零零的标题，看起来像是加载失败
        binding.titleSessions.visibility = if (state.sessions.isEmpty()) View.GONE else View.VISIBLE

        val container = binding.containerSessions
        container.removeAllViews()
        for (session in state.sessions) {
            val item = ItemHistorySessionBinding.inflate(layoutInflater, container, false)
            val label = if (session.source == PracticeSource.REVIEW) {
                "${gradeName(session.grade)} ${getString(R.string.history_session_review_tag)}"
            } else {
                gradeName(session.grade)
            }
            item.textSession.text = getString(
                R.string.history_session_format,
                timeFormat.format(Date(session.createdAt)), label, session.correct, session.total,
            )
            container.addView(item.root)
        }
    }

    private fun renderMistakes(state: HistoryState) {
        binding.titleMistakes.visibility = if (state.mistakes.isEmpty()) View.GONE else View.VISIBLE

        val container = binding.containerMistakes
        container.removeAllViews()
        for (mistake in state.mistakes) {
            val item = ItemMistakeBinding.inflate(layoutInflater, container, false)
            item.textMistakeProblem.text = getString(
                R.string.history_mistake_format, mistake.problem.expression, mistake.problem.answer,
            )
            item.textMistakeWritten.text = getString(
                R.string.history_mistake_written_format,
                mistake.written.ifEmpty { getString(R.string.math_review_unrecognized) },
                mistake.mistakeCount,
            )
            container.addView(item.root)
        }

        // 超出的部分要明说，否则学生会以为错题丢了
        val beyond = state.mistakesBeyondReviewLimit
        binding.textMistakesBeyond.visibility = if (beyond > 0) View.VISIBLE else View.GONE
        if (beyond > 0) {
            binding.textMistakesBeyond.text = getString(R.string.history_mistakes_beyond_format, beyond)
        }

        // 错题本为空就不给入口 —— 点了也不会有反应
        binding.buttonReviewMistakes.isEnabled = state.mistakes.isNotEmpty()
    }

    /** 复用 Spinner 那份年级名单。顺序与 `Grade` 枚举一致由 `DigitInputTest` 之外的约定钉住。 */
    private fun gradeName(grade: Grade): String =
        resources.getStringArray(R.array.grade_names)[grade.ordinal]
}
