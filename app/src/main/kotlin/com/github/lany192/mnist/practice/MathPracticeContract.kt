package com.github.lany192.mnist.practice

import com.github.lany192.mnist.recognize.DigitInput
import com.github.lany192.mnist.recognize.digitsText

/** 单题作答记录。错题回顾直接遍历这个列表，不再另存一份错题。 */
data class Attempt(
    /** 1-based，用于"第 N 题"的展示。 */
    val index: Int,
    val problem: Problem,
    /** 学生写出的数字位。存 `List<Int>` 而不是 String，与 [DigitInput.Digits.value] 保持一致，省掉一次 join/split 往返。 */
    val written: List<Int>,
    val correct: Boolean,
    /** 小数点位于第几个数字之前；空表示整数。 */
    val decimalIndexes: List<Int> = emptyList(),
) {
    /** 拼回学生实际写出的文本，保留前导零和小数点。 */
    val writtenText: String get() = digitsText(written, decimalIndexes)
}

/**
 * 题目从哪来。
 *
 * 它同时决定三件事：年级选择器是否可用、进度文案怎么写、以及这次练习是否计入累计正确率。
 */
sealed interface ProblemSource {
    /** 按年级现出的题。 */
    data object ByGrade : ProblemSource

    /** 错题重做。题目可能跨年级，所以年级没有语义 —— 界面上隐藏选择器。 */
    data object ReviewMistakes : ProblemSource
}

/**
 * 作答阶段。
 *
 * 用 sealed interface 带载荷，而不是 enum + 若干个必须手工保持同步的并列字段：识别结果
 * 只在"待确认/已判定"里有意义，把它绑在分支上之后，"阶段已回到作答中、结果字段还留着上一题的"
 * 这类错位在结构上就不可能发生 —— 旧实现要靠三处手工重置才能维持一致。
 */
sealed interface MathPracticePhase {
    data object Answering : MathPracticePhase

    data class Confirming(
        val digits: List<Int>,
        val decimalIndexes: List<Int> = emptyList(),
    ) : MathPracticePhase

    data class Judged(
        val digits: List<Int>,
        val decimalIndexes: List<Int> = emptyList(),
    ) : MathPracticePhase

    data object Finished : MathPracticePhase
}

data class MathPracticeState(
    /** 年级选择器的当前值。重做态下界面不显示它，但仍保留最后一次选择用于归档。 */
    val grade: Grade = Grade.FIRST,
    val source: ProblemSource = ProblemSource.ByGrade,
    val problems: List<Problem> = emptyList(),
    val index: Int = 0,
    val attempts: List<Attempt> = emptyList(),
    val phase: MathPracticePhase = MathPracticePhase.Answering,
) {
    /** 是否处于错题重做态。界面据此隐藏年级选择器、切换进度与结算文案。 */
    val isReviewing: Boolean get() = source is ProblemSource.ReviewMistakes

    /**
     * Finished 停在**最后一题**上、而不是把 index 推到 problems.size，这样 [currentProblem]
     * 永远合法。旧实现在结算后 index 已越界，只是靠"结算分支不索引 problems"侥幸不崩。
     */
    val currentProblem: Problem get() = problems[index]

    val problemCount: Int get() = problems.size

    val isLastProblem: Boolean get() = index >= problems.lastIndex

    val correctCount: Int get() = attempts.count { it.correct }

    val wrongAttempts: List<Attempt> get() = attempts.filterNot { it.correct }
}

sealed interface MathPracticeIntent {
    data class SelectGrade(val grade: Grade) : MathPracticeIntent

    /** 点了「提交」。[input] 是 View 采集并识别出的原始结果。 */
    data class Submit(val input: DigitInput) : MathPracticeIntent

    /** 点了「重写」：回到作答态并清空画布。 */
    data object Rewrite : MathPracticeIntent

    /** 点了「确认」：对已识别出的结果做判定。 */
    data object Confirm : MathPracticeIntent

    /** 点了「下一题」。 */
    data object Next : MathPracticeIntent

    /** 点了「再来一组」。重做态下重跑同一批错题，不退回按年级出题。 */
    data object StartNewSet : MathPracticeIntent

    /**
     * 拿这批题练一遍（错题重做）。
     *
     * **空列表会被忽略**：错题本为空、或跨页传递的两个数组长度不匹配，都可能在界面上
     * 留下一个可点的入口。如果放行，`problems[index]` 会直接越界崩溃。
     */
    data class StartReview(val problems: List<Problem>) : MathPracticeIntent

    /**
     * 拿**本次练错的题**再练一遍（结算页的按钮）。
     *
     * 与 [StartReview] 分开是为了区分两件事：这个来自用户点击，每次都要生效；
     * [StartReview] 来自跨页 Intent 的灌注，重复灌注意味着 Activity 被重建，必须幂等。
     */
    data object ReviewCurrentMistakes : MathPracticeIntent

    /** 点了「清除」：只清画布，不改阶段。 */
    data object ClearCanvas : MathPracticeIntent
}

/**
 * 一次性事件。
 *
 * 规则：**凡是新建的 Activity 必须能复现的东西，一律进 [MathPracticeState]，不进这里**。
 * 画布冻结就是反例 —— 它是 `phase is Answering` 的纯函数，由 View 从状态推导，
 * 一旦做成 Effect，旋转后新 Activity 的画布默认可写、而状态仍是"待确认"，两边就撕裂了。
 *
 * 文案由 View 决定，ViewModel 不引用 `R.string`。
 */
sealed interface MathPracticeEffect {
    data object ClearCanvas : MathPracticeEffect

    /** 画布空白就点了提交。 */
    data object EmptyCanvas : MathPracticeEffect

    /** 有笔迹，但切不出任何数字。 */
    data object NotRecognized : MathPracticeEffect

    /** 识别到的数字位数超过上限，只取前 [max] 位。 */
    data class TooManyDigits(val max: Int) : MathPracticeEffect

    /** 小数点位置非法，例如多个点、点在开头或末尾。 */
    data object InvalidNumber : MathPracticeEffect
}
