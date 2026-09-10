package com.github.lany192.mnist

import android.graphics.Bitmap

/**
 * 整条识别链路的编排：切分 → 逐位归一化 → 逐位推理 → 拼成数值。
 *
 * 抽成独立类是为了让仪器测试能直接跑**与 App 完全相同**的代码路径：
 * 切分算法有 JVM 单测覆盖，但 Bitmap 缩放、TFLite 推理这两段只能在 Android 运行时里验证。
 */
class MnistRecognizer(private val tflite: KerasTFLite) {

    class Result(val digits: List<Int>, val totalCount: Int)

    /**
     * @param maxDigits 最多识别几位，超出的部分被截断，但 [Result.totalCount] 会如实反映实际切出的位数。
     */
    fun recognize(bitmap: Bitmap, maxDigits: Int): Result {
        val boxes = segment(bitmap)
        val digits = ArrayList<Int>(minOf(boxes.size, maxDigits))
        for (box in boxes.take(maxDigits)) {
            val input = MnistPreprocessor.preprocessRegion(bitmap, box)
            if (input.isEmpty()) continue
            val digit = tflite.classify(input)
            if (digit in 0..9) digits.add(digit)
        }
        return Result(digits, boxes.size)
    }

    /** 导出整张画布的墨迹并切分，返回按书写顺序排列的各个数字的紧包围盒。 */
    fun segment(bitmap: Bitmap): List<Box> {
        val width = bitmap.width
        val height = bitmap.height
        if (width <= 0 || height <= 0) return emptyList()
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        return InkSegmenter.segment(InkSegmenter.toInkGray(pixels), width, height)
    }
}
