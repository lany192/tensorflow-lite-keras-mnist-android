package com.github.lany192.mnist

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.github.lany192.mnist.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

/**
 * 单数字识别页。MVI：View 只做三件事 —— 把输入翻译成 [MainIntent]、订阅 [MainState] 渲染、
 * 执行一次性 [MainEffect]。所有状态转移在 [MainViewModel] 里，可在 JVM 上单测。
 */
class MainActivity : AppCompatActivity() {
    private var mTFLite: KerasTFLite? = null
    private var mRecognizer: MnistRecognizer? = null
    private lateinit var binding: ActivityMainBinding
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.textHint.text = getString(R.string.hint_write, MAX_RECOGNIZED_DIGITS)
        binding.buttonDetect.setOnClickListener { onSubmit() }
        binding.buttonClear.setOnClickListener { viewModel.dispatch(MainIntent.Clear) }
        binding.buttonMathPractice.setOnClickListener {
            startActivity(Intent(this, MathPracticeActivity::class.java))
        }
        binding.buttonHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }
        val tflite = KerasTFLite(this)
        mTFLite = tflite
        mRecognizer = MnistRecognizer(tflite)

        // 先同步渲染一次，避免 onStart 之前出现一帧空白
        render(viewModel.state.value)
        lifecycleScope.launch {
            // 用 repeatOnLifecycle 而不是裸 launch：停止期间不渲染、不弹 Toast。
            // 代价是每次 onStart 会重放一次当前状态，所以 render 必须幂等
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { viewModel.state.collect(::render) }
                launch { viewModel.effect.collect(::handleEffect) }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // 在 onPause 释放会让 App 切后台再回来时解释器已关闭，识别直接抛异常
        mTFLite?.release()
        mTFLite = null
        mRecognizer = null
    }

    private fun onSubmit() {
        val activeRecognizer = mRecognizer
        if (activeRecognizer == null) {
            // 既存行为：识别器不可用时复用了"请先写一个数字"的文案，语义并不准确。
            // 本次重构只改架构、不改可观察行为，要改走单独一个 commit。
            showToast(getString(R.string.toast_empty))
            return
        }
        viewModel.dispatch(MainIntent.Submit(collectDigitInput(activeRecognizer)))
    }

    /**
     * 只做「采集 + 翻译」，不决定任何策略：空画布怎么办、切不出数字怎么办、位数超了要不要提示，
     * 全部交给 [MainViewModel] 判定 —— 那三条是唯一有分支的路径，放在这里就永远测不到。
     *
     * 识别刻意留在 View 层：`MnistRecognizer` 吃 `Bitmap`，ViewModel 一旦依赖 Bitmap 就再也
     * 进不了 JVM 单测（项目没有 Robolectric）。
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

    /** 幂等：`repeatOnLifecycle` 每次 onStart 都会重放一次当前状态。 */
    private fun render(state: MainState) {
        if (!state.hasResult) {
            binding.textResult.text = ""
            binding.textDetail.text = ""
            return
        }
        binding.textResult.text = getString(R.string.result_format, state.value)
        // 逐位拆分用于区分"切分错了"还是"认错了"，实机排查时是关键信息
        binding.textDetail.text = getString(
            R.string.digit_detail_format,
            state.glyphCount,
            state.value.toCharArray().joinToString(" ")
        )
    }

    private fun handleEffect(effect: MainEffect) {
        when (effect) {
            MainEffect.ClearCanvas -> binding.fingerPaintView.clear()
            MainEffect.EmptyCanvas -> showToast(getString(R.string.toast_empty))
            MainEffect.NotRecognized -> showToast(getString(R.string.toast_no_ink))
            is MainEffect.TooManyDigits -> showToast(getString(R.string.toast_too_many, effect.max))
            MainEffect.InvalidNumber -> showToast(getString(R.string.toast_invalid_decimal))
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
