package com.github.lany192.mnist

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow

/**
 * 单数字识别页的状态机。
 *
 * 与 [MathPracticeViewModel] 遵循同一套 MVI 约定（见那个类的注释，那里写了为什么：
 * 内部保持同步、Effect 只装一次性事件、状态转移全部收敛在 [reduce] 里）。
 *
 * **不要在这个文件里 import 任何 `android.*`** —— `PureKotlinBoundaryTest` 会拦下来。
 */
class MainViewModel(
    private val maxDigits: Int = MAX_RECOGNIZED_DIGITS,
) : ViewModel() {

    private val _state = MutableStateFlow(MainState())
    val state: StateFlow<MainState> = _state.asStateFlow()

    // UNLIMITED：Effect 是必须送达的一次性事件，无界队列让 trySend 结构性地不会失败
    private val _effect = Channel<MainEffect>(Channel.UNLIMITED)
    val effect: Flow<MainEffect> = _effect.receiveAsFlow()

    fun dispatch(intent: MainIntent) {
        val (next, effects) = reduce(intent)
        _state.value = next
        effects.forEach { _effect.trySend(it) }
    }

    private data class Transition(
        val state: MainState,
        val effects: List<MainEffect> = emptyList(),
    )

    /** 所有状态转移只在这一个函数里发生，其余方法不得直接写 `_state.value`。 */
    private fun reduce(intent: MainIntent): Transition {
        val current = _state.value
        return when (intent) {
            MainIntent.Clear -> Transition(MainState(), listOf(MainEffect.ClearCanvas))

            is MainIntent.Submit -> when (val input = intent.input) {
                // 两种"没拿到数字"都保持原状态：旧结果不该被一次失败的识别擦掉
                DigitInput.CanvasEmpty -> Transition(current, listOf(MainEffect.EmptyCanvas))
                DigitInput.NotRecognized -> Transition(current, listOf(MainEffect.NotRecognized))
                is DigitInput.Digits -> Transition(
                    current.copy(digits = input.value),
                    // 超位数只截断不阻断，与识别器"截断但如实报告实际位数"的语义一致
                    if (input.totalCount > maxDigits) {
                        listOf(MainEffect.TooManyDigits(maxDigits))
                    } else {
                        emptyList()
                    }
                )
            }
        }
    }
}
