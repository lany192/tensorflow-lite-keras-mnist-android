package com.github.lany192.mnist.data

import com.github.lany192.mnist.practice.Attempt
import com.github.lany192.mnist.practice.Grade
import com.github.lany192.mnist.practice.MathPracticeViewModel
import com.github.lany192.mnist.practice.Problem
import kotlinx.coroutines.flow.Flow

/**
 * 持久化边界。
 *
 * 这个文件是**纯 Kotlin**（`kotlinx.coroutines.flow.Flow` 是纯 JVM 类，不违反
 * `PureKotlinBoundaryTest` 的 `android.*` 约束）。把接口留在这里、把 Room 实现留在
 * `RoomPracticeRepository` 换来的东西是具体的：KSP 若哪天走不通、或要换成别的存储，
 * 只需要换掉实现类那一个文件，ViewModel、Contract、测试、界面全都不用动。
 */

/** 一次练习的来源。决定它是否计入累计正确率。 */
enum class PracticeSource {
    /** 按年级出的题。计入累计正确率。 */
    GRADE,

    /**
     * 错题重做。**照样写库**（错题本正是靠"最新一条记录"自愈的），
     * 但**不计入正确率** —— 刚看过答案的题参与统计，这个数字就没有意义了。
     */
    REVIEW,
}

/**
 * 待归档的一次练习。由 `MathPracticeViewModel` 的 reduce **产出**（因此 reduce 仍是纯函数），
 * 再由 [PracticeRecorder] 落库。
 */
data class PracticeArchive(
    /** 发起这次练习时选中的年级。重做态的题目可能跨年级，所以这里只是"当时的选择"，不追求精确。 */
    val grade: Grade,
    val source: PracticeSource,
    val attempts: List<Attempt>,
)

/**
 * 练习归档端口。
 *
 * **[record] 不得阻塞调用线程、不得抛异常。** 它是在 `dispatch` 的返回路径上被调用的，
 * 那一刻学生的结算页已经渲染完毕 —— 归档出问题绝不能让练习流程出问题。
 * 实现方负责立刻返回，把真正的 I/O 交给应用级作用域。
 *
 * 为什么不让 Activity 观察状态、进入结算时自己写库：`configChanges` 已经移除，旋转会重建
 * Activity —— 结算瞬间旋转会让 `lifecycleScope` 被取消、写入夭折，而 Effect 已经被消费、
 * 不会重发。结果是结算页正常显示、数据一条没写、没有任何报错。
 */
fun interface PracticeRecorder {
    fun record(archive: PracticeArchive)

    companion object {
        /** 默认实现：什么都不做。存在的唯一意义是让 ViewModel 的无参构造保持可用。 */
        val NoOp = PracticeRecorder { }
    }
}

/** 按年级的累计正确率。 */
data class GradeStats(val grade: Grade, val correct: Int, val total: Int) {
    /** 正确率百分比。`total` 为 0 时返回 null，让界面决定怎么表达"还没有数据"，而不是显示成 0%。 */
    val percent: Int? get() = if (total == 0) null else correct * 100 / total
}

/** 历史练习列表的一行。 */
data class SessionSummary(
    val id: Long,
    val grade: Grade,
    val source: PracticeSource,
    /** epoch 毫秒。展示层负责格式化（用 `SimpleDateFormat`，`java.time` 需要 API 26+）。 */
    val createdAt: Long,
    val total: Int,
    val correct: Int,
)

/** 错题本的一条。 */
data class MistakeRecord(
    val problem: Problem,
    /** 学生最近一次写错的内容。 */
    val written: String,
    /** 同一道题累计错了几次，用于提示"这题错了 3 次"。 */
    val mistakeCount: Int,
)

/**
 * 只读访问。写入走 [PracticeRecorder] —— 因为**写不可重推、读可以重推**：
 * Room 的 Flow 每次订阅都会重发，所以读可以由 View 层收集后灌进 ViewModel；写没有第二次机会。
 */
interface PracticeRepository {
    fun observeGradeStats(): Flow<List<GradeStats>>

    fun observeSessions(): Flow<List<SessionSummary>>

    fun observeMistakes(): Flow<List<MistakeRecord>>
}
