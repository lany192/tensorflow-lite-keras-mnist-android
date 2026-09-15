package com.github.lany192.mnist.history

import androidx.lifecycle.ViewModel
import com.github.lany192.mnist.practice.MathPracticeViewModel
import com.github.lany192.mnist.recognize.MainViewModel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow

/**
 * 学习记录页的状态机。
 *
 * 与 [MainViewModel] 一样是**无参构造 + `by viewModels()` + 纯 reducer** —— 因为它的数据
 * （累计正确率、历史练习、错题本）都是"可以重新推导"的：Room 的 Flow 每次订阅都会重发，
 * 所以由 View 层收集后 `dispatch(Loaded(...))` 灌进来即可，不需要把数据库注入进来。
 *
 * 对比 [MathPracticeViewModel]：那里需要注入 recorder，因为**写不可重推**。
 * "写不可重推、读可以重推"这条分界决定了两个页面为什么长成不同的样子。
 *
 * **不要在这个文件里 import 任何 `android.*`** —— `PureKotlinBoundaryTest` 会拦下来。
 */
class HistoryViewModel : ViewModel() {

    private val _state = MutableStateFlow(HistoryState())
    val state: StateFlow<HistoryState> = _state.asStateFlow()

    private val _effect = Channel<HistoryEffect>(Channel.UNLIMITED)
    val effect: Flow<HistoryEffect> = _effect.receiveAsFlow()

    fun dispatch(intent: HistoryIntent) {
        val (next, effects) = reduce(intent)
        _state.value = next
        effects.forEach { _effect.trySend(it) }
    }

    private data class Transition(
        val state: HistoryState,
        val effects: List<HistoryEffect> = emptyList(),
    )

    private fun reduce(intent: HistoryIntent): Transition = when (intent) {
        is HistoryIntent.Loaded -> Transition(
            HistoryState(
                stats = intent.stats,
                sessions = intent.sessions,
                mistakes = intent.mistakes,
            )
        )

        // 错题本为空时不发 Effect：界面上已经挡住了入口，这是可单测的第二道防线。
        // 放行的话练习页会收到一个空题目集，而它已经在状态层拒绝这种输入（转而不做任何事），
        // 结果就是一个"点了没反应"的按钮。
        HistoryIntent.ReviewMistakes -> {
            val problems = reviewProblemsOf(_state.value.mistakes)
            if (problems.isEmpty()) Transition(_state.value) else Transition(_state.value, listOf(HistoryEffect.StartReview(problems)))
        }
    }
}
