package com.github.lany192.mnist

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.github.lany192.mnist.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private var mTFLite: KerasTFLite? = null
    private var mRecognizer: MnistRecognizer? = null
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.textHint.text = getString(R.string.hint_write, MAX_DIGITS)
        binding.buttonDetect.setOnClickListener { onDetectClicked() }
        binding.buttonClear.setOnClickListener { onClearClicked() }
        val tflite = KerasTFLite(this)
        mTFLite = tflite
        mRecognizer = MnistRecognizer(tflite)
    }

    override fun onDestroy() {
        super.onDestroy()
        // 在 onPause 释放会让 App 切后台再回来时解释器已关闭，识别直接抛异常
        mTFLite?.release()
        mTFLite = null
        mRecognizer = null
    }

    private fun onDetectClicked() {
        if (binding.fingerPaintView.isEmpty) {
            showToast(getString(R.string.toast_empty))
            return
        }
        val recognizer = mRecognizer
        if (recognizer == null) {
            showToast(getString(R.string.toast_empty))
            return
        }
        val bitmap = binding.fingerPaintView.exportDrawingBitmap()
        try {
            val result = recognizer.recognize(bitmap, MAX_DIGITS)
            if (result.digits.isEmpty()) {
                showToast(getString(R.string.toast_no_ink))
                return
            }
            val value = result.digits.joinToString("")
            binding.textResult.text = getString(R.string.result_format, value)
            // 逐位拆分用于区分"切分错了"还是"认错了"，实机排查时是关键信息
            binding.textDetail.text = getString(
                R.string.digit_detail_format,
                result.digits.size,
                value.toCharArray().joinToString(" ")
            )
            if (result.totalCount > MAX_DIGITS) {
                showToast(getString(R.string.toast_too_many, MAX_DIGITS))
            }
        } finally {
            bitmap.recycle()
        }
    }

    private fun onClearClicked() {
        binding.fingerPaintView.clear()
        binding.textResult.text = ""
        binding.textDetail.text = ""
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    companion object {
        /**
         * 实测整串全对率随位数下降（32px 笔宽：2 位 93%、3 位 77%、4 位 87%、5 位 60%），
         * 位数再多每个数字就会被写小，"笔画宽 / 数字高度"越过悬崖，结果必然出错。
         * 与其让用户拿到一个必然是错的结果，不如明确截断并提示。
         */
        private const val MAX_DIGITS = 4
    }
}
