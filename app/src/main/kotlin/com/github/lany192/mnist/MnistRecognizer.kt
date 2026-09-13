package com.github.lany192.mnist

import android.graphics.Bitmap

/**
 * 整条识别链路的编排：切分 → 逐字形归一化 → 逐字形推理 → 拼成数值。
 *
 * 数字和小数点都经过模型：小数点样本由 Python 训练脚本合成，归一化后是实心椭圆；
 * 切分器只负责保留“形状和位置都可能是小数点”的候选，最终类别仍由 11 类模型决定。
 * 这样误触圆点即使侥幸通过切分，也不会被直接当成小数点。
 */
class MnistRecognizer(private val tflite: KerasTFLite) {

    /**
     * @param digits 已识别的数字，可能因 [maxDigits] 被截断。
     * @param totalCount 切分算法实际切出的数字总数，可能大于 [digits] 的长度。
     * @param decimalIndexes 小数点位于第几个数字之前；合法小数应恰好有一个元素，
     *   且值位于 `1 until digits.size`。例如 `digits = [0, 2]`、`decimalIndexes = [1]`
     *   表示 `"0.2"`。
     */
    class Result(
        val digits: List<Int>,
        val totalCount: Int,
        val decimalIndexes: List<Int> = emptyList(),
    )

    /**
     * @param maxDigits 最多识别几位数字，超出的部分被截断，但 [Result.totalCount] 会如实反映实际切出的数字总数。
     */
    fun recognize(bitmap: Bitmap, maxDigits: Int): Result {
        val glyphs = segmentGlyphs(bitmap)
        val digits = ArrayList<Int>(minOf(glyphs.size, maxDigits))
        val decimalIndexes = ArrayList<Int>()
        var totalDigitCount = 0
        var pendingDots = 0
        // 切分器只负责把“可能是点”的小块从噪声过滤中救下来，不再决定它最终是什么。
        // 这样把一个写得很小的数字 2/3/5 误判成点候选，仍能被 11 类模型纠正回数字；
        // 反之，普通数字框若被模型判成点，也会按小数点处理。
        fun addClassified(symbol: Int) {
            when (symbol) {
                KerasTFLite.DECIMAL_POINT_CLASS -> {
                    // 只有已经写出前导数字、且后面还留得下数字时，点才可能是小数点。
                    if (digits.isNotEmpty() && digits.size < maxDigits) pendingDots++
                }

                in 0..9 -> {
                    totalDigitCount++
                    if (digits.size >= maxDigits) {
                        // 已截断：后面的点不再有数字可以依附，直接丢弃。
                        pendingDots = 0
                        return
                    }
                    repeat(pendingDots) { decimalIndexes.add(digits.size) }
                    pendingDots = 0
                    digits.add(symbol)
                }
            }
        }

        for (glyph in glyphs) {
            val input = MnistPreprocessor.preprocessRegion(bitmap, glyph.box)
            if (input.isEmpty()) continue
            addClassified(tflite.classify(input))
        }
        return Result(digits, totalDigitCount, decimalIndexes)
    }

    /** 导出整张画布并切分；此兼容入口只返回数字框。 */
    fun segment(bitmap: Bitmap): List<Box> = segmentGlyphs(bitmap)
        .filter { it.kind == GlyphKind.DIGIT }
        .map { it.box }

    /** 导出整张画布并切分，返回按书写顺序排列的数字与小数点字形。 */
    fun segmentGlyphs(bitmap: Bitmap): List<InkGlyph> {
        val width = bitmap.width
        val height = bitmap.height
        if (width <= 0 || height <= 0) return emptyList()
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        return InkSegmenter.segmentGlyphs(InkSegmenter.toInkGray(pixels), width, height)
    }
}
