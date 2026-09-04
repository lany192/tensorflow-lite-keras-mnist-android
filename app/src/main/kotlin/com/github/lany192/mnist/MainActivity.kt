package com.github.lany192.mnist

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.github.lany192.mnist.databinding.ActivityMainBinding
import kotlin.math.max
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    private var mTFLite: KerasTFLite? = null
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.buttonDetect.setOnClickListener { onDetectClicked() }
        binding.buttonClear.setOnClickListener { onClearClicked() }
        mTFLite = KerasTFLite(this)
        binding.buttonDetect.visibility = android.view.View.VISIBLE
    }

    override fun onPause() {
        super.onPause()
        mTFLite?.release()
    }

    private fun onDetectClicked() {
        if (binding.fingerPaintView.isEmpty) {
            Toast.makeText(this, "请写上一个数字", Toast.LENGTH_SHORT).show()
            return
        }
        val bitmap = binding.fingerPaintView.exportDrawingBitmap()
        val pixels = preprocess(bitmap)
        bitmap.recycle()
        if (pixels.isEmpty()) {
            Toast.makeText(this, "请写上一个数字", Toast.LENGTH_SHORT).show()
            return
        }
        binding.textResult.text = "数字是: ${mTFLite?.run(pixels)}"
    }

    private fun onClearClicked() {
        binding.fingerPaintView.clear()
        binding.textResult.text = ""
    }

    /**
     * MNIST 风格预处理：裁剪笔画包围盒 -> 等比缩放到 20px -> 亮度归一化 -> 按质心居中到 28x28。
     * 输出 float 0~1，笔画为 1，与训练分布一致。
     */
    private fun preprocess(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        // 笔画亮度 0~255（白底黑字，取反后笔画为亮）
        val gray = FloatArray(w * h)
        for (i in pixels.indices) {
            gray[i] = (0xff - (pixels[i] and 0xff)).toFloat()
        }
        // 包围盒
        var minX = w
        var minY = h
        var maxX = -1
        var maxY = -1
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (gray[y * w + x] > INK_THRESHOLD) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        if (maxX < 0) return FloatArray(0)
        val crop = Bitmap.createBitmap(bitmap, minX, minY, maxX - minX + 1, maxY - minY + 1)
        // 大幅缩小前多级减半，避免双线性采样跳过细笔画；全程保持宽高比
        var current = crop
        while (maxOf(current.width, current.height) > MID_SIZE * 2) {
            val half = Bitmap.createScaledBitmap(
                current,
                max(1, current.width / 2),
                max(1, current.height / 2),
                true
            )
            if (half == current) break
            if (current != crop) current.recycle()
            current = half
        }
        val midScale = MID_SIZE.toFloat() / maxOf(current.width, current.height)
        val mid = Bitmap.createScaledBitmap(
            current,
            max(1, (current.width * midScale).roundToInt()),
            max(1, (current.height * midScale).roundToInt()),
            true
        )
        val scale = CONTENT_SIZE.toFloat() / maxOf(mid.width, mid.height)
        val dw = max(1, (mid.width * scale).roundToInt())
        val dh = max(1, (mid.height * scale).roundToInt())
        val scaled = Bitmap.createScaledBitmap(mid, dw, dh, true)
        mid.recycle()
        if (current != crop) current.recycle()
        crop.recycle()
        // 缩放后笔画可能变淡，按最大亮度归一化到 255
        val sPixels = IntArray(dw * dh)
        scaled.getPixels(sPixels, 0, dw, 0, 0, dw, dh)
        var maxInk = 0f
        for (i in sPixels.indices) {
            val ink = (0xff - (sPixels[i] and 0xff)).toFloat()
            if (ink > maxInk) maxInk = ink
        }
        val brightness = if (maxInk > 0f) 255f / maxInk else 1f
        // 先按包围盒居中绘制，计算质心
        val inkPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        val centered = Bitmap.createBitmap(MODEL_SIZE, MODEL_SIZE, Bitmap.Config.ARGB_8888)
        val centeredCanvas = Canvas(centered)
        centeredCanvas.drawColor(Color.WHITE)
        centeredCanvas.drawBitmap(scaled, (MODEL_SIZE - dw) / 2f, (MODEL_SIZE - dh) / 2f, inkPaint)
        val cPixels = IntArray(MODEL_SIZE * MODEL_SIZE)
        centered.getPixels(cPixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)
        var sumX = 0f
        var sumY = 0f
        var sumW = 0f
        for (y in 0 until MODEL_SIZE) {
            for (x in 0 until MODEL_SIZE) {
                val ink = (0xff - (cPixels[y * MODEL_SIZE + x] and 0xff)).toFloat() * brightness
                if (ink > 0f) {
                    sumX += ink * x
                    sumY += ink * y
                    sumW += ink
                }
            }
        }
        centered.recycle()
        // 按质心平移到画布中心（与 MNIST 预处理一致）
        var dx = 0f
        var dy = 0f
        if (sumW > 0f) {
            dx = MODEL_SIZE / 2f - sumX / sumW
            dy = MODEL_SIZE / 2f - sumY / sumW
        }
        val result = Bitmap.createBitmap(MODEL_SIZE, MODEL_SIZE, Bitmap.Config.ARGB_8888)
        val resultCanvas = Canvas(result)
        resultCanvas.drawColor(Color.WHITE)
        resultCanvas.drawBitmap(
            scaled,
            (MODEL_SIZE - dw) / 2f + dx,
            (MODEL_SIZE - dh) / 2f + dy,
            inkPaint
        )
        scaled.recycle()
        val rPixels = IntArray(MODEL_SIZE * MODEL_SIZE)
        result.getPixels(rPixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)
        result.recycle()
        val out = FloatArray(MODEL_SIZE * MODEL_SIZE)
        for (i in rPixels.indices) {
            val ink = (0xff - (rPixels[i] and 0xff)).toFloat() * brightness
            out[i] = (ink.coerceIn(0f, 255f)) / 255f
        }
        return out
    }

    companion object {
        private const val MID_SIZE = 80
        private const val MODEL_SIZE = 28
        private const val CONTENT_SIZE = 20
        private const val INK_THRESHOLD = 32f
    }
}
