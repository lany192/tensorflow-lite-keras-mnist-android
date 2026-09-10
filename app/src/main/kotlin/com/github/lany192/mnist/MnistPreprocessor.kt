package com.github.lany192.mnist

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * MNIST 风格的输入归一化。
 *
 * 数值行为必须与历史单数字版本保持逐字节一致：这里的每一步（多级减半缩放、
 * 缩放到 [MID_SIZE]、缩放到 [CONTENT_SIZE]、亮度归一化、按质心居中）都是为了让
 * 手指绘制出的粗笔画贴合 MNIST 的分布，任何"顺手优化"都会破坏识别效果。
 * 该类依赖 Android 的 Bitmap 缩放实现（无法在 JVM 上复刻），因此切分逻辑不要写在这里。
 */
object MnistPreprocessor {
    const val MODEL_SIZE = 28
    private const val MID_SIZE = 80
    private const val CONTENT_SIZE = 20

    /**
     * 把 [box] 区域归一化成 28x28 的 784 个 float（笔画为 1，背景为 0）。
     *
     * [box] 必须是该数字的紧墨迹包围盒，不要为了"保险"加 padding：
     * 缩放分母和质心计算都依赖这个紧包围盒，加 padding 会让数字在 28x28 里变小，
     * 同时破坏与单数字版本的等价性。
     *
     * 单个数字时 box 即全图墨迹包围盒，输出与改造前完全一致。
     * 区域无墨时返回长度为 0 的数组。
     */
    fun preprocessRegion(bitmap: Bitmap, box: Box): FloatArray {
        if (box.width <= 0 || box.height <= 0) return FloatArray(0)
        val crop = Bitmap.createBitmap(bitmap, box.left, box.top, box.width, box.height)
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
        // 缩放后笔画可能变淡，按最大亮度归一化到 255。
        // 这里是逐个数字独立归一化：若跨数字共享一个全局最亮值，某一位写得重
        // 就会让其余数字的笔画达不到饱和，输入偏离训练分布。
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
}
