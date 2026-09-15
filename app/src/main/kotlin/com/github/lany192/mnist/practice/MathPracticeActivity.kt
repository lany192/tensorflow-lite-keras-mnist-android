package com.github.lany192.mnist.practice

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.MenuItem
import android.view.View
import android.widget.AdapterView
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.github.lany192.mnist.R
import com.github.lany192.mnist.data.PracticeDatabase
import com.github.lany192.mnist.databinding.ActivityMathPracticeBinding
import com.github.lany192.mnist.recognize.DigitInput
import com.github.lany192.mnist.recognize.KerasTFLite
import com.github.lany192.mnist.recognize.MAX_RECOGNIZED_DIGITS
import com.github.lany192.mnist.recognize.MainActivity
import com.github.lany192.mnist.recognize.MnistRecognizer
import com.github.lany192.mnist.recognize.digitInputOf
import com.github.lany192.mnist.recognize.digitsText
import kotlinx.coroutines.launch

/**
 * 小学口算练习页。MVI：View 只做三件事 —— 把输入翻译成 [MathPracticeIntent]、订阅
 * [MathPracticeState] 渲染、执行一次性 [MathPracticeEffect]。所有状态转移在
 * [MathPracticeViewModel] 里，由 `MathPracticeViewModelTest` 的 21 条用例覆盖。
 *
 * 识别刻意留在这一层：`MnistRecognizer` 吃 `Bitmap`，ViewModel 一旦依赖 Bitmap 就再也进不了
 * JVM 单测（项目没有 Robolectric）。这里只做「采集 + 翻译」成 [DigitInput]，不决定任何策略。
 *
 * 本 Activity 与 MainActivity 会**各自持有一个 TFLite Interpreter**（MainActivity 只是被
 * stop，并未 destroy）。MNIST CNN 模型很小，这是有意为之而非泄漏；两边都只在 onDestroy 释放。
 */
class MathPracticeActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMathPracticeBinding
    private var tflite: KerasTFLite? = null
    private var recognizer: MnistRecognizer? = null
    private val reviewAdapter = ReviewAdapter()

    private val viewModel: MathPracticeViewModel by viewModels {
        // 用显式工厂而不是无参构造，是因为 recorder 必须从外面注入 —— 让 ViewModel 自己去
        // 全局拿，会让"忘了初始化"表现为"一切正常但一条数据都不写"。
        MathPracticeViewModel.factory(PracticeDatabase.recorder(applicationContext))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMathPracticeBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 二级页统一的返回入口：ActionBar 左上角的返回箭头（见 DESIGN.md 1.3）。
        // 刻意不放进页面内容区 —— 首行已经被「年级 + 进度」占满，往里加控件会撑高那一行、
        // 挤矮 weight=1 的画布，触发 onSizeChanged 重建位图并擦掉学生刚写的字迹。
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        // 错题回顾列表。这三件必须在第一次 render 之前装配完：少任何一件，RecyclerView 都只是
        // 静默地不显示内容。它整体在 groupResult 内（作答态 GONE），所以不参与画布的高度计算。
        binding.recyclerReview.layoutManager = LinearLayoutManager(this)
        binding.recyclerReview.adapter = reviewAdapter
        // 原来是一个 TextView 拼多行，本来就没有动画；显式关掉才是忠实迁移。
        binding.recyclerReview.itemAnimator = null

        val interpreter = KerasTFLite(this)
        tflite = interpreter
        recognizer = MnistRecognizer(interpreter)
        bindActions()

        // 从错题本进来时带着题目。**只在首次创建时灌注** —— 旋转会重建 Activity，而 ViewModel
        // 还活着、重做的题目与进度都在，再灌一次会把进度重置（ViewModel 里也有第二道闸门）。
        if (savedInstanceState == null) applyReviewFromIntent(intent)

        // 先同步渲染一次，避免 onStart 之前出现一帧空白。这一步**不会改变任何可见性** ——
        // 布局 XML 里的默认值与"作答态"的渲染结果逐项相同；若哪天不一致，这次渲染会挤矮画布、
        // 触发 onSizeChanged 清空笔迹（详见布局里 textCheckHint 的注释）。
        render(viewModel.state.value)
        lifecycleScope.launch {
            // 用 repeatOnLifecycle 而不是裸 launch：停止期间不渲染、不弹 Toast。
            // 代价是每次 onStart 会重放一次当前状态，所以 render 必须幂等且不得清画布。
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { viewModel.state.collect(::render) }
                launch { viewModel.effect.collect(::handleEffect) }
            }
        }
    }

    /**
     * ActionBar 左上角的返回箭头。与系统返回键同效：中途退出不归档，
     * 已完成但未收尾的那组练习会丢失（既有取舍，见 CLAUDE.md「持久化」）。
     */
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == android.R.id.home) {
            finish()
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    override fun onDestroy() {
        super.onDestroy()
        // 与 MainActivity 同因：不能在 onPause 释放，否则切后台再回来解释器已关闭，识别直接抛异常
        tflite?.release()
        tflite = null
        recognizer = null
    }

    private fun bindActions() {
        binding.spinnerGrade.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                // 首次布局会带 position=0 回调一次，render 回写 selection 时也会回调；
                // 两种情况都由 ViewModel 里"同年级即无操作"的闸门挡掉，不会白出一轮题。
                viewModel.dispatch(MathPracticeIntent.SelectGrade(Grade.values()[position]))
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        binding.buttonClear.setOnClickListener { viewModel.dispatch(MathPracticeIntent.ClearCanvas) }
        binding.buttonSubmit.setOnClickListener { onSubmit() }
        binding.buttonRewrite.setOnClickListener { viewModel.dispatch(MathPracticeIntent.Rewrite) }
        binding.buttonConfirm.setOnClickListener { viewModel.dispatch(MathPracticeIntent.Confirm) }
        binding.buttonNext.setOnClickListener { viewModel.dispatch(MathPracticeIntent.Next) }
        binding.buttonNextSet.setOnClickListener { viewModel.dispatch(MathPracticeIntent.StartNewSet) }
        binding.buttonReviewMistakes.setOnClickListener {
            viewModel.dispatch(MathPracticeIntent.ReviewCurrentMistakes)
        }
    }

    /**
     * 还原跨页传来的错题。没有 extra 说明这次是普通的按年级练习，直接返回。
     *
     * 两个数组长度不匹配时**结束页面**：带着一个残缺的题目集进去，`problems[index]` 会越界。
     */
    private fun applyReviewFromIntent(intent: Intent) {
        val expressions = intent.getStringArrayListExtra(EXTRA_REVIEW_EXPRESSIONS) ?: return
        val answers = intent.getIntegerArrayListExtra(EXTRA_REVIEW_ANSWERS) ?: return
        val problems = problemsOf(expressions, answers)
        if (problems.isNullOrEmpty()) {
            // 两个数组长度不匹配（或都是空的）：与其带着残缺的题目集进去让 problems[index] 越界，
            // 不如直接结束页面。
            finish()
            return
        }
        viewModel.dispatch(MathPracticeIntent.StartReview(problems))
    }

    private fun onSubmit() {
        // 识别器不可用时静默返回（既存行为，本次重构只改架构、不改可观察行为）
        val activeRecognizer = recognizer ?: return
        viewModel.dispatch(MathPracticeIntent.Submit(collectDigitInput(activeRecognizer)))
    }

    /**
     * 只做「采集 + 翻译」，不决定任何策略：空画布怎么办、切不出数字怎么办、位数超了要不要提示，
     * 全部交给 ViewModel —— 那三条是唯一有分支的路径，放在这一层就永远测不到。
     */
    private fun collectDigitInput(recognizer: MnistRecognizer): DigitInput {
        if (binding.fingerPaintView.isEmpty) return DigitInput.CanvasEmpty
        val bitmap = binding.fingerPaintView.exportDrawingBitmap()
        val result = try {
            recognizer.recognize(bitmap, MAX_RECOGNIZED_DIGITS)
        } finally {
            bitmap.recycle()
        }
        return digitInputOf(
            canvasEmpty = false,
            digits = result.digits,
            totalCount = result.totalCount,
            decimalIndexes = result.decimalIndexes,
        )
    }

    /**
     * 唯一的视图状态出口。必须**幂等**，因为 `repeatOnLifecycle` 每次 onStart 都会重放一次
     * 当前状态；也**不得在这里 `dispatch()`**，否则会与 Spinner 的回调形成回环。
     */
    private fun render(state: MathPracticeState) {
        val phase = state.phase
        val answering = phase is MathPracticePhase.Answering
        val confirming = phase is MathPracticePhase.Confirming
        val judged = phase is MathPracticePhase.Judged
        val finished = phase is MathPracticePhase.Finished

        binding.groupQuestion.visibility = if (finished) View.GONE else View.VISIBLE
        binding.groupResult.visibility = if (finished) View.VISIBLE else View.GONE

        binding.buttonClear.visibility = if (answering) View.VISIBLE else View.GONE
        binding.buttonSubmit.visibility = if (answering) View.VISIBLE else View.GONE
        binding.buttonRewrite.visibility = if (confirming) View.VISIBLE else View.GONE
        binding.buttonConfirm.visibility = if (confirming) View.VISIBLE else View.GONE
        binding.buttonNext.visibility = if (judged) View.VISIBLE else View.GONE
        binding.buttonNextSet.visibility = if (finished) View.VISIBLE else View.GONE

        // 用 INVISIBLE 而非 GONE：这行一旦从"不占位"变成"占位"，画布会被挤矮，
        // onSizeChanged 重建位图就会把学生刚写的字迹清掉。详见布局里的注释。
        binding.textCheckHint.visibility = if (confirming) View.VISIBLE else View.INVISIBLE

        // 画布冻结由阶段推导，**不能走 Effect**：Effect 不跨配置变更存活，靠 Effect 冻结会让
        // "旋转后状态仍是待确认、画布却变成可写的"——学生以为锁着、实际能乱写，
        // 再点「确认」判定的还是刚才那个数。当前行为恰好等价：只有作答态可写。
        binding.fingerPaintView.inputEnabled = answering

        // 重做态的题目跨年级，年级选择器没有语义。
        // **必须是 INVISIBLE，绝不能用 GONE** —— 这一行是 wrap_content，它变矮会挤高 weight=1
        // 的画布、触发 onSizeChanged 重建位图、把学生刚写的字迹清掉（AGENTS.md 记着这个事故）。
        val showGrade = !state.isReviewing
        binding.textGradeLabel.visibility = if (showGrade) View.VISIBLE else View.INVISIBLE
        binding.spinnerGrade.visibility = if (showGrade) View.VISIBLE else View.INVISIBLE
        binding.spinnerGrade.isEnabled = showGrade
        // Spinner 是有状态控件、自己也存着一份选中位置，是第二个真相来源：只在必要时回写，
        // 回写触发的回调会被 ViewModel 里"同年级即无操作"的闸门吸收，不形成回环。
        // 重做态下完全不碰它 —— 那时的 grade 只是"发起重做时选的年级"，回写没有意义。
        if (showGrade && binding.spinnerGrade.selectedItemPosition != state.grade.ordinal) {
            binding.spinnerGrade.setSelection(state.grade.ordinal)
        }

        binding.textProgress.text = when {
            // 结算页已经没有"第几题"的语义了
            finished -> ""
            state.isReviewing ->
                getString(R.string.math_review_progress_format, state.index + 1, state.problemCount)

            else -> getString(R.string.math_progress_format, state.index + 1, state.problemCount)
        }
        binding.textExpression.text = getString(R.string.math_expression_format, state.currentProblem.expression)

        // 反馈文字的颜色与它的文案同源，因此不可能出现「文案说答错、颜色说答对」。
        // 默认中性：确认态写的"识别结果：X"只是回显，学生此刻正是要核对它，本身不含对错。
        binding.textFeedback.setTextColor(getColor(R.color.on_surface))
        when (phase) {
            is MathPracticePhase.Confirming -> {
                val text = digitsText(phase.digits, phase.decimalIndexes)
                binding.textFeedback.text = getString(R.string.result_format, text)
                // 复用 MainActivity 的逐字形拆分：得让学生看出是哪一位认错了，才能决定要不要重写。
                // 4 位答案的全对率只有 87%，这一步是防误判的主要关口。
                binding.textDetail.text = getString(
                    R.string.digit_detail_format,
                    phase.digits.size + phase.decimalIndexes.size,
                    text.toCharArray().joinToString(" ")
                )
            }

            is MathPracticePhase.Judged -> {
                // Judged 态由 ViewModel 保证只可能是答错（答对会直接 advance 到下一题），
                // 所以这里的错误色与下面那句"回答错误"文案用的是同一个前提，不会漂移。
                binding.textFeedback.setTextColor(getColor(R.color.wrong))
                binding.textFeedback.text =
                    getString(R.string.math_wrong_answer_format, state.currentProblem.answer)
                binding.textDetail.text = getString(
                    R.string.math_your_answer_format,
                    digitsText(phase.digits, phase.decimalIndexes)
                )
            }

            // 进入作答态必须清掉上一次的结果，否则点"重写"后旧的"识别结果：1"会残留在屏幕上
            MathPracticePhase.Answering -> {
                binding.textFeedback.text = ""
                binding.textDetail.text = ""
            }

            MathPracticePhase.Finished -> Unit
        }

        if (finished) {
            // 与下面的 when 用的是同一个条件，所以颜色跟着文案走：
            // 只有"全部答对 N 题，太棒了！"那一句才上绿色。重做模式下只复述成绩，不提"太棒了"。
            val allCorrect = !state.isReviewing && state.wrongAttempts.isEmpty()
            binding.textSummary.setTextColor(
                getColor(if (allCorrect) R.color.correct else R.color.on_surface)
            )
            binding.textSummary.text = when {
                state.isReviewing ->
                    getString(R.string.math_review_summary_format, state.correctCount, state.attempts.size)

                state.wrongAttempts.isEmpty() ->
                    getString(R.string.math_summary_all_correct, state.attempts.size)

                else ->
                    getString(R.string.math_summary_format, state.correctCount, state.attempts.size)
            }
            // 没有错题就不给这个入口 —— 点了也是空操作
            binding.buttonReviewMistakes.visibility =
                if (state.wrongAttempts.isNotEmpty()) View.VISIBLE else View.GONE
        }

        // **无条件提交**，不要加 `if (state.wrongAttempts.isNotEmpty())` 之类的守卫：
        // 结算后点「再来一组」会清空 attempts，作答完成后 groupResult 重新可见，
        // 守卫会让上一组的错题行原样留在适配器里 —— 按钮已正确隐藏，列表却还在。
        reviewAdapter.submitList(if (finished) state.wrongAttempts else emptyList())
    }

    private fun handleEffect(effect: MathPracticeEffect) {
        when (effect) {
            MathPracticeEffect.ClearCanvas -> binding.fingerPaintView.clear()
            MathPracticeEffect.EmptyCanvas -> showToast(getString(R.string.math_toast_write_first))
            MathPracticeEffect.NotRecognized -> showToast(getString(R.string.math_toast_detect_failed))
            is MathPracticeEffect.TooManyDigits -> showToast(getString(R.string.toast_too_many, effect.max))
            MathPracticeEffect.InvalidNumber -> showToast(getString(R.string.math_toast_invalid_decimal))
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    companion object {
        private const val EXTRA_REVIEW_EXPRESSIONS = "review_expressions"
        private const val EXTRA_REVIEW_ANSWERS = "review_answers"

        /**
         * 用一批题目启动练习页（错题重做）。
         *
         * 题目用**两个平行数组**传，而不是让 `Problem` 实现 `Parcelable`：`android.os.Parcel`
         * 是 `android.*`，会让 `MathProblem.kt` 掉出纯 Kotlin 白名单，连带它的单测一起失去
         * JVM 可测性。数组在 `onCreate` 里同步还原，所以没有"先渲染帧再替换"的闪烁。
         */
        fun reviewIntent(context: Context, problems: List<Problem>): Intent =
            Intent(context, MathPracticeActivity::class.java)
                .putStringArrayListExtra(EXTRA_REVIEW_EXPRESSIONS, ArrayList(problems.map { it.expression }))
                .putIntegerArrayListExtra(EXTRA_REVIEW_ANSWERS, ArrayList(problems.map { it.answer }))
    }
}
