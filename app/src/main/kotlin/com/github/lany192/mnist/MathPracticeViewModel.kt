package com.github.lany192.mnist

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow

/**
 * 小学口算练习的状态机。
 *
 * MVI 约定（两个界面的 ViewModel 都遵守）：
 *
 * 1. **内部刻意保持同步**。`MutableStateFlow.value =` 与 `Channel.trySend` 都不挂起，所以
 *    `dispatch()` 返回时状态已经就位，单测直接断言 `state.value` 即可，**不需要
 *    `kotlinx-coroutines-test` / `TestDispatcher`**。更硬的理由是 `Dispatchers.Main` 在 JVM 单测里
 *    不可用（`coroutines-android` 是 Android 产物）—— 一旦这里出现 `viewModelScope.launch`，
 *    测试会以 `Module with the Main dispatcher had failed to initialize` 失败，而不是编译失败。
 *
 * 2. **所有状态转移只在 [reduce] 一个函数里发生**，其余方法只允许返回 [Transition]，不得直接写
 *    `_state.value`。这样将来真需要异步时，把 [reduce] 整体搬成纯函数就是机械操作。
 *
 * 3. **Effect 只装一次性事件**。凡是新建的 Activity 必须能复现的东西，一律进 [MathPracticeState]。
 *
 * **不要在这个文件里 import 任何 `android.*`** —— `PureKotlinBoundaryTest` 会拦下来。
 */
class MathPracticeViewModel(
    /**
     * 出题函数可注入，使测试能拿到完全确定的题目。
     * 与 [maxDigits] 一样带默认值，Kotlin 才会生成无参构造，`by viewModels()` 才能用 ——
     * 一旦有人删掉默认值，运行时报 `Cannot create an instance of class`，编译期毫无提示。
     */
    private val generate: (Grade) -> List<Problem> = { MathProblemGenerator.generateSet(it) },
    private val maxDigits: Int = MAX_RECOGNIZED_DIGITS,
    /**
     * 练习归档端口。
     *
     * 新增参数**一律要给默认值**：现有测试用的是具名参数构造，去掉默认值会让它们直接编译失败。
     */
    private val recorder: PracticeRecorder = PracticeRecorder.NoOp,
) : ViewModel() {

    private val _state = MutableStateFlow(MathPracticeState(problems = generate(Grade.FIRST)))
    val state: StateFlow<MathPracticeState> = _state.asStateFlow()

    // UNLIMITED：Effect 是必须送达的一次性事件，无界队列让 trySend 结构性地不会失败
    private val _effect = Channel<MathPracticeEffect>(Channel.UNLIMITED)
    val effect: Flow<MathPracticeEffect> = _effect.receiveAsFlow()

    fun dispatch(intent: MathPracticeIntent) {
        val (next, effects, archive) = reduce(intent)
        // 先落状态：结算页不等待任何 I/O
        _state.value = next
        effects.forEach { _effect.trySend(it) }
        // 再交付归档。不挂起、不返回 —— 实现在内部把真正的写库交给应用级作用域，
        // 所以结算瞬间旋转屏幕也不会丢掉这次记录。放在 Activity 的 lifecycleScope 里就会：
        // 协程被取消、而 Effect 已经消费不会重发，结果是页面正常、数据没写、毫无报错。
        archive?.let(recorder::record)
    }

    private data class Transition(
        val state: MathPracticeState,
        val effects: List<MathPracticeEffect> = emptyList(),
        /**
         * 需要归档的一次练习。[reduce] 只**产出**它、不执行 —— 这样 reduce 依然是纯函数，
         * 将来要把它整体搬成纯函数也还是机械操作。
         */
        val archive: PracticeArchive? = null,
    )

    private fun reduce(intent: MathPracticeIntent): Transition {
        val current = _state.value
        return when (intent) {
            // 同年级不动：Spinner 首次布局会用 position=0 回调一次，render 回写 selection 时也会
            // 回调。这道闸门是防 render→dispatch→render 回环的关键，不能省。
            // 但重做态下选年级意味着"退出重做、回到按年级出题"，所以那时即使同年级也要重建。
            is MathPracticeIntent.SelectGrade ->
                if (intent.grade == current.grade && !current.isReviewing) {
                    Transition(current)
                } else {
                    newSet(intent.grade)
                }

            // 重做态下「再来一组」= 重跑同一批错题，不退回按年级出题
            MathPracticeIntent.StartNewSet ->
                if (current.isReviewing) restartReview(current) else newSet(current.grade)

            // 来自跨页 Intent 的灌注。已是重做态说明 Activity 被重建（旋转），必须幂等 ——
            // 否则学生的重做进度会被重置。
            is MathPracticeIntent.StartReview ->
                if (current.isReviewing) Transition(current) else beginReview(current, intent.problems)

            // 来自结算页的按钮，每次点击都要生效（包括重做之后再重做）
            MathPracticeIntent.ReviewCurrentMistakes ->
                beginReview(current, current.wrongAttempts.map { it.problem })

            // 非对应阶段收到这两条一律忽略：按钮可见性已经挡住了，这里是第二道防线，
            // 让"重复点击"在状态层不可能造成破坏。
            MathPracticeIntent.Confirm ->
                if (current.phase is MathPracticePhase.Confirming) confirm(current) else Transition(current)

            MathPracticeIntent.Next ->
                if (current.phase is MathPracticePhase.Judged) advance(current) else Transition(current)

            MathPracticeIntent.Rewrite -> Transition(
                current.copy(phase = MathPracticePhase.Answering),
                listOf(MathPracticeEffect.ClearCanvas)
            )

            MathPracticeIntent.ClearCanvas -> Transition(current, listOf(MathPracticeEffect.ClearCanvas))

            is MathPracticeIntent.Submit -> submit(current, intent.input)
        }
    }

    /**
     * 空画布与"切不出数字"都必须**留在作答态**：若进了待确认，「确认」会把一个空结果判成错，
     * 等于凭空冤枉学生。
     */
    private fun submit(current: MathPracticeState, input: DigitInput): Transition = when (input) {
        DigitInput.CanvasEmpty -> Transition(current, listOf(MathPracticeEffect.EmptyCanvas))
        DigitInput.NotRecognized -> Transition(current, listOf(MathPracticeEffect.NotRecognized))
        is DigitInput.Digits -> Transition(
            current.copy(phase = MathPracticePhase.Confirming(input.value)),
            // 超位数仍然进待确认：只截断不阻断，与识别器"截断但如实报告实际位数"的语义一致
            if (input.totalCount > maxDigits) {
                listOf(MathPracticeEffect.TooManyDigits(maxDigits))
            } else {
                emptyList()
            }
        )
    }

    private fun confirm(current: MathPracticeState): Transition {
        val digits = (current.phase as MathPracticePhase.Confirming).digits
        // 必须按数值比较：学生写 "068" 与答案 68 是同一个数，字符串比较会把它判错，
        // 而带前导零的写法在手写时很常见。
        val written = digits.joinToString("").toIntOrNull()
        val correct = written != null && written == current.currentProblem.answer
        val recorded = current.copy(
            attempts = current.attempts + Attempt(current.index + 1, current.currentProblem, digits, correct)
        )
        // 答对不停顿，直接进入下一题；答错停在已判定态等学生看完再点「下一题」。
        return if (correct) advance(recorded) else Transition(recorded.copy(phase = MathPracticePhase.Judged(digits)))
    }

    private fun advance(current: MathPracticeState): Transition =
        if (current.isLastProblem) {
            // 整批归档（单事务）。中途退出不写 —— 已知取舍：做 9 题退出会丢掉这 9 题的记录，
            // 但"练习"的语义是完整一组，半场数据会污染正确率的分母。
            Transition(
                current.copy(phase = MathPracticePhase.Finished),
                archive = PracticeArchive(
                    grade = current.grade,
                    source = if (current.isReviewing) PracticeSource.REVIEW else PracticeSource.GRADE,
                    attempts = current.attempts,
                ),
            )
        } else {
            Transition(
                current.copy(index = current.index + 1, phase = MathPracticePhase.Answering),
                listOf(MathPracticeEffect.ClearCanvas)
            )
        }

    /** 出新一组题并把作答态清零。切换年级与「再来一组」走同一条路径，避免两处重置逻辑各自漂移。 */
    private fun newSet(grade: Grade): Transition = Transition(
        // MathPracticeState 的默认 source 就是 ByGrade，所以这里天然退出重做态
        MathPracticeState(grade = grade, problems = generate(grade)),
        listOf(MathPracticeEffect.ClearCanvas)
    )

    /**
     * 进入错题重做态。
     *
     * **空列表直接忽略**：错题本为空、或跨页传递的两个数组长度不匹配，都可能在界面上留下一个
     * 可点的入口；放行会让 `problems[index]` 越界崩溃。
     */
    private fun beginReview(current: MathPracticeState, problems: List<Problem>): Transition =
        if (problems.isEmpty()) {
            Transition(current)
        } else {
            Transition(
                current.copy(
                    source = ProblemSource.ReviewMistakes,
                    problems = problems,
                    index = 0,
                    attempts = emptyList(),
                    phase = MathPracticePhase.Answering,
                ),
                listOf(MathPracticeEffect.ClearCanvas),
            )
        }

    /** 重做态下的「再来一组」：重跑**同一批**错题，而不是退回按年级出题。 */
    private fun restartReview(current: MathPracticeState): Transition = Transition(
        current.copy(index = 0, attempts = emptyList(), phase = MathPracticePhase.Answering),
        listOf(MathPracticeEffect.ClearCanvas)
    )

    companion object {
        /**
         * 显式工厂（`ViewModelProvider` 是 `androidx.*`，不影响本文件的纯 Kotlin 边界）。
         *
         * **故意不提供"从全局拿 recorder"的默认实现** —— 那会让"忘了初始化"表现为
         * "一切正常但一条数据都不写"，是项目最讨厌的那类静默失败。
         */
        fun factory(recorder: PracticeRecorder): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T =
                    MathPracticeViewModel(recorder = recorder) as T
            }
    }
}
