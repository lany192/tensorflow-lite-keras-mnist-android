package com.github.lany192.mnist

import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.github.lany192.mnist.databinding.ActivityMathPracticeBinding
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
    private val viewModel: MathPracticeViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMathPracticeBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val interpreter = KerasTFLite(this)
        tflite = interpreter
        recognizer = MnistRecognizer(interpreter)
        bindActions()

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
        return digitInputOf(canvasEmpty = false, digits = result.digits, totalCount = result.totalCount)
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

        // Spinner 是有状态控件、自己也存着一份选中位置，是第二个真相来源：只在必要时回写，
        // 回写触发的回调会被 ViewModel 里"同年级即无操作"的闸门吸收，不形成回环。
        if (binding.spinnerGrade.selectedItemPosition != state.grade.ordinal) {
            binding.spinnerGrade.setSelection(state.grade.ordinal)
        }

        binding.textProgress.text = if (finished) {
            // 结算页已经没有"第几题"的语义了
            ""
        } else {
            getString(R.string.math_progress_format, state.index + 1, state.problemCount)
        }
        binding.textExpression.text = getString(R.string.math_expression_format, state.currentProblem.expression)

        when (phase) {
            is MathPracticePhase.Confirming -> {
                binding.textFeedback.text = getString(R.string.result_format, phase.digits.joinToString(""))
                // 复用 MainActivity 的逐位拆分：得让学生看出是哪一位认错了，才能决定要不要重写。
                // 4 位答案的全对率只有 87%，这一步是防误判的主要关口。
                binding.textDetail.text = getString(
                    R.string.digit_detail_format,
                    phase.digits.size,
                    phase.digits.joinToString(" ")
                )
            }

            is MathPracticePhase.Judged -> {
                binding.textFeedback.text =
                    getString(R.string.math_wrong_answer_format, state.currentProblem.answer)
                binding.textDetail.text = getString(R.string.math_your_answer_format, phase.digits.joinToString(""))
            }

            // 进入作答态必须清掉上一次的结果，否则点"重写"后旧的"识别结果：1"会残留在屏幕上
            MathPracticePhase.Answering -> {
                binding.textFeedback.text = ""
                binding.textDetail.text = ""
            }

            MathPracticePhase.Finished -> Unit
        }

        if (finished) {
            binding.textSummary.text = if (state.wrongAttempts.isEmpty()) {
                getString(R.string.math_summary_all_correct, state.attempts.size)
            } else {
                getString(R.string.math_summary_format, state.correctCount, state.attempts.size)
            }
            binding.textReview.text = state.wrongAttempts.joinToString("\n") { attempt ->
                getString(
                    R.string.math_review_line_format,
                    attempt.index,
                    attempt.problem.expression,
                    attempt.problem.answer,
                    // 当前不可达：进 Confirming 的必要条件就是识别出至少一位数字，故 written 必非空。
                    // 保留它是为了不改动已验证的文案结构；要清理请单独开一个 commit（那是行为变更）。
                    attempt.written.joinToString("").ifEmpty { getString(R.string.math_review_unrecognized) }
                )
            }
        }
    }

    private fun handleEffect(effect: MathPracticeEffect) {
        when (effect) {
            MathPracticeEffect.ClearCanvas -> binding.fingerPaintView.clear()
            MathPracticeEffect.EmptyCanvas -> showToast(getString(R.string.math_toast_write_first))
            MathPracticeEffect.NotRecognized -> showToast(getString(R.string.math_toast_detect_failed))
            is MathPracticeEffect.TooManyDigits -> showToast(getString(R.string.toast_too_many, effect.max))
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
