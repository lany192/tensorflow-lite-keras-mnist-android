package com.github.lany192.mnist

import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.github.lany192.mnist.databinding.ActivityMathPracticeBinding

/**
 * 小学口算练习：逐题作答，手写答案后提交识别、确认、判定。
 *
 * 完全复用 [FingerPaintView] / [MnistRecognizer] / [KerasTFLite]，识别链路一行未改。
 *
 * **为什么要两次点击**：手写识别有出错概率（实测 3 位整串全对率 77%、4 位 87%），若"提交"即
 * 判定，学生写对了却被认错就直接被判错。所以第一次点击只做识别、把结果显示出来让学生核对，
 * 确认无误再点"确认"才判定；若识别错了可以点"重写"。
 *
 * 本 Activity 与 MainActivity 会**各自持有一个 TFLite Interpreter**（MainActivity 只是被 stop，
 * 并未 destroy）。MNIST CNN 模型很小，这是有意为之而非泄漏；两边都只在 onDestroy 释放，
 * 原因见 MainActivity 里同一处注释（放 onPause 会让切后台再回来时解释器已关闭）。
 */
class MathPracticeActivity : AppCompatActivity() {

    private enum class State {
        /** 作答中：等学生写完并点"提交"。 */
        ANSWERING,

        /** 待确认：已识别出结果，等学生核对后点"确认"或"重写"。 */
        CONFIRMING,

        /** 答错后停住：显示正确答案，等学生看完点"下一题"。 */
        JUDGED,

        /** 一组题做完，显示成绩与错题回顾。 */
        FINISHED
    }

    /** 单题作答记录。错题回顾直接遍历这个列表，不再另存一份错题。 */
    private data class Attempt(
        val index: Int,
        val problem: Problem,
        val written: String,
        val correct: Boolean
    )

    private lateinit var binding: ActivityMathPracticeBinding
    private var tflite: KerasTFLite? = null
    private var recognizer: MnistRecognizer? = null

    private var grade = Grade.FIRST
    private var problems: List<Problem> = emptyList()
    private var index = 0
    private val attempts = mutableListOf<Attempt>()
    private var state = State.ANSWERING

    /** 提交时冻结的识别结果。判定的是"学生看到的那个数"，而不是画布此刻的内容。 */
    private var recognizedText = ""
    private var recognizedDigits: List<Int> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMathPracticeBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val interpreter = KerasTFLite(this)
        tflite = interpreter
        recognizer = MnistRecognizer(interpreter)
        bindActions()
        startSet(grade)
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
                val selected = Grade.values()[position]
                // Spinner 首次布局时就会用 position=0 回调一次，此时 onCreate 已经出好题了；
                // 靠"与当前年级相同就跳过"把它挡掉，否则会白跑一轮出题
                if (selected == grade) return
                startSet(selected)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        binding.buttonClear.setOnClickListener { binding.fingerPaintView.clear() }
        binding.buttonSubmit.setOnClickListener { onSubmit() }
        binding.buttonRewrite.setOnClickListener { onRewrite() }
        binding.buttonConfirm.setOnClickListener { onConfirm() }
        binding.buttonNext.setOnClickListener { advance() }
        binding.buttonNextSet.setOnClickListener { startSet(grade) }
    }

    /** 出新一组题并把作答态清零。切换年级与"再来一组"走同一条路径，避免两处重置逻辑各自漂移。 */
    private fun startSet(selected: Grade) {
        grade = selected
        problems = MathProblemGenerator.generateSet(grade)
        index = 0
        attempts.clear()
        showProblem()
    }

    private fun showProblem() {
        binding.textProgress.text = getString(R.string.math_progress_format, index + 1, problems.size)
        binding.textExpression.text = getString(R.string.math_expression_format, problems[index].expression)
        binding.fingerPaintView.clear()
        binding.fingerPaintView.inputEnabled = true
        recognizedText = ""
        recognizedDigits = emptyList()
        state = State.ANSWERING
        render()
    }

    private fun onSubmit() {
        if (binding.fingerPaintView.isEmpty) {
            showToast(getString(R.string.math_toast_write_first))
            return
        }
        val activeRecognizer = recognizer ?: return
        val bitmap = binding.fingerPaintView.exportDrawingBitmap()
        val result = try {
            activeRecognizer.recognize(bitmap, MathProblemGenerator.ANSWER_DIGITS)
        } finally {
            bitmap.recycle()
        }
        if (result.digits.isEmpty()) {
            // 只点一下画布会被实心块过滤器滤掉、笔画写得太小也切不出数字。此时必须留在作答态：
            // 若进了确认态，"确认"会把一个空结果判成错，等于凭空冤枉学生
            showToast(getString(R.string.math_toast_detect_failed))
            return
        }
        recognizedText = result.digits.joinToString("")
        recognizedDigits = result.digits
        if (result.totalCount > MathProblemGenerator.ANSWER_DIGITS) {
            showToast(getString(R.string.toast_too_many, MathProblemGenerator.ANSWER_DIGITS))
        }
        // 确认的就是屏幕上这个数，画布不该再被改动
        binding.fingerPaintView.inputEnabled = false
        state = State.CONFIRMING
        render()
    }

    private fun onRewrite() {
        binding.fingerPaintView.clear()
        binding.fingerPaintView.inputEnabled = true
        recognizedText = ""
        recognizedDigits = emptyList()
        state = State.ANSWERING
        render()
    }

    private fun onConfirm() {
        val problem = problems[index]
        // 必须按数值比较而非字符串比较：学生写 "068" 与答案 68 是同一个数，
        // 字符串比较会把它判错，而带前导零的写法在手写时很常见
        val written = recognizedText.toIntOrNull()
        val correct = written != null && written == problem.answer
        attempts.add(Attempt(index + 1, problem, recognizedText, correct))
        if (correct) {
            // 答对不停顿，直接进入下一题
            advance()
        } else {
            state = State.JUDGED
            render()
        }
    }

    /** 进入下一题；已经是最后一题则结算。 */
    private fun advance() {
        index++
        if (index < problems.size) showProblem() else finishSet()
    }

    private fun finishSet() {
        val correctCount = attempts.count { it.correct }
        val wrong = attempts.filterNot { it.correct }
        binding.textSummary.text = if (wrong.isEmpty()) {
            getString(R.string.math_summary_all_correct, attempts.size)
        } else {
            getString(R.string.math_summary_format, correctCount, attempts.size)
        }
        binding.textReview.text = wrong.joinToString("\n") { attempt ->
            getString(
                R.string.math_review_line_format,
                attempt.index,
                attempt.problem.expression,
                attempt.problem.answer,
                attempt.written.ifEmpty { getString(R.string.math_review_unrecognized) }
            )
        }
        state = State.FINISHED
        render()
    }

    /** 所有可见性与文案切换只在这一个函数里发生，避免状态和界面在多个回调里各自漂移。 */
    private fun render() {
        val answering = state == State.ANSWERING
        val confirming = state == State.CONFIRMING
        val judged = state == State.JUDGED
        val finished = state == State.FINISHED

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

        when (state) {
            State.CONFIRMING -> {
                binding.textFeedback.text = getString(R.string.result_format, recognizedText)
                // 复用 MainActivity 的逐位拆分：得让学生看出是哪一位认错了，才能决定要不要重写。
                // 4 位答案的全对率只有 87%，这一步是防误判的主要关口
                binding.textDetail.text = getString(
                    R.string.digit_detail_format,
                    recognizedDigits.size,
                    recognizedDigits.joinToString(" ")
                )
            }

            State.JUDGED -> {
                binding.textFeedback.text =
                    getString(R.string.math_wrong_answer_format, problems[index].answer)
                binding.textDetail.text = getString(R.string.math_your_answer_format, recognizedText)
            }

            // 进入作答态必须清掉上一次的结果，否则点"重写"后旧的"识别结果：1"会残留在屏幕上
            State.ANSWERING -> {
                binding.textFeedback.text = ""
                binding.textDetail.text = ""
            }

            // 结算页已经没有"第几题"的语义了，清掉进度显示
            State.FINISHED -> binding.textProgress.text = ""
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
