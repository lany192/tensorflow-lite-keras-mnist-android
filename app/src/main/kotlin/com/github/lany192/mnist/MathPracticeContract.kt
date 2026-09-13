package com.github.lany192.mnist

/** 单题作答记录。错题回顾直接遍历这个列表，不再另存一份错题。 */
data class Attempt(
    /** 1-based，用于"第 N 题"的展示。 */
    val index: Int,
    val problem: Problem,
    /** 学生写出的数字位。存 `List<Int>` 而不是 String，与 [DigitInput.Digits.value] 保持一致，省掉一次 join/split 往返。 */
    val written: List<Int>,
    val correct: Boolean,
)

/**
 * 作答阶段。
 *
 * 用 sealed interface 带载荷，而不是 enum + 若干个必须手工保持同步的并列字段：识别结果
 * 只在"待确认/已判定"里有意义，把它绑在分支上之后，"阶段已回到作答中、结果字段还留着上一题的"
 * 这类错位在结构上就不可能发生 —— 旧实现要靠三处手工重置才能维持一致。
 */
sealed interface MathPracticePhase {
    data object Answering : MathPracticePhase

    data class Confirming(val digits: List<Int>) : MathPracticePhase

    data class Judged(val digits: List<Int>) : MathPracticePhase

    data object Finished : MathPracticePhase
}

data class MathPracticeState(
    val grade: Grade = Grade.FIRST,
    val problems: List<Problem> = emptyList(),
    val index: Int = 0,
    val attempts: List<Attempt> = emptyList(),
    val phase: MathPracticePhase = MathPracticePhase.Answering,
) {
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

    /** 点了「再来一组」。 */
    data object StartNewSet : MathPracticeIntent

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

    /** 识别到的位数超过上限，只取前 [max] 位。 */
    data class TooManyDigits(val max: Int) : MathPracticeEffect
}
