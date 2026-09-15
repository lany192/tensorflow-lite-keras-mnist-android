package com.github.lany192.mnist

import android.os.Bundle
import android.view.MenuItem
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.github.lany192.mnist.databinding.ActivityHistoryBinding
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch

/**
 * 学习记录页。MVI：View 订阅 Room 的 Flow 后灌进 [HistoryViewModel]，再按状态渲染界面。
 *
 * 数据流方向与练习页**相反**是刻意的 —— 判据是"这次操作的结果可不可以重新推导"：
 * **读可以重推**（Flow 每次订阅都会重发），所以读取由 View 层收集即可，这个 ViewModel
 * 保持无参构造、纯 reducer；而**写不可重推**，所以练习页必须把归档端口注入进 ViewModel。
 *
 * 本页整页是一个 `RecyclerView`：把状态摊平成行列表这件事在 [historyRowsOf] 里，是纯 Kotlin；
 * 这里只负责装配适配器、提交行列表，以及那个不属于列表的底部按钮。
 */
class HistoryActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryBinding
    private val viewModel: HistoryViewModel by viewModels()
    private val adapter = HistoryAdapter()

    /** 上一次提交给适配器的状态，用来把重复的 render 变成空操作。 */
    private var renderedState: HistoryState? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 这三件必须在第一次 render 之前装配完。少任何一件 RecyclerView 都只是**静默地不显示内容**
        // （没有 layoutManager 时它连布局都不会走），而状态恢复紧跟在 onCreate 之后。
        binding.recyclerHistory.layoutManager = LinearLayoutManager(this)
        binding.recyclerHistory.adapter = adapter
        // 现有实现是 removeAllViews + addView，本来就没有动画；显式关掉才是忠实迁移，
        // 也让 DiffUtil 的结果直接生效，不掺一层淡入淡出。
        binding.recyclerHistory.itemAnimator = null

        val repository = PracticeDatabase.repository(applicationContext)

        // 二级页统一的返回入口：ActionBar 左上角的返回箭头（见 DESIGN.md 1.3）
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
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

    /** ActionBar 左上角的返回箭头。finish() 回到首页，与系统返回键同效。 */
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == android.R.id.home) {
            finish()
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    private fun handleEffect(effect: HistoryEffect) {
        when (effect) {
            is HistoryEffect.StartReview ->
                startActivity(MathPracticeActivity.reviewIntent(this, effect.problems))
        }
    }

    /**
     * 幂等 —— 每次 onStart 都会重放。所以先比状态再提交：`submitList` 会把 diff 丢到后台线程算，
     * 状态没变时那次计算纯属白费。比的是 [HistoryState] 而不是行列表，因为状态才是真正的输入。
     */
    private fun render(state: HistoryState) {
        if (state != renderedState) {
            renderedState = state
            adapter.submitList(state.rows)
        }

        // 错题本为空就不给入口 —— 点了也不会有反应
        binding.buttonReviewMistakes.isEnabled = state.mistakes.isNotEmpty()
    }
}
