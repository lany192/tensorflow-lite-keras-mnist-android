package com.github.lany192.mnist

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 端到端仪器测试：跑的是与 App 完全相同的链路（切分 → Bitmap 归一化 → TFLite 推理）。
 *
 * 切分算法已有 JVM 单测覆盖；这里补的是 JVM 测不到的两段——`Bitmap.createScaledBitmap` 的
 * 真实缩放行为，以及 TFLite 解释器加载 assets/model.tflite 后的真实推理。
 * 因此断言以**位数和流程稳健性**为主，对具体数值只做有把握的断言：
 * 手绘的几何图形与 MNIST 的字迹分布有差异，对数值断言过严会让测试变得脆弱。
 */
@RunWith(AndroidJUnit4::class)
class MnistRecognizerTest {

    private lateinit var tflite: KerasTFLite
    private lateinit var recognizer: MnistRecognizer

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        tflite = KerasTFLite(context)
        recognizer = MnistRecognizer(tflite)
    }

    @After
    fun tearDown() {
        tflite.release()
    }

    @Test
    fun emptyCanvas_recognizesNothing() {
        val result = recognizer.recognize(blankCanvas(), NO_LIMIT)
        assertTrue("空画布不应切出任何数字，实际 ${result.digits}", result.digits.isEmpty())
        assertEquals(0, result.totalCount)
    }

    @Test
    fun singleDot_recognizesNothing() {
        val bitmap = blankCanvas()
        val canvas = Canvas(bitmap)
        val dots = Paint().apply { color = Color.BLACK; isAntiAlias = true }
        canvas.drawCircle(300f, 300f, STROKE_WIDTH / 2f, dots)
        val result = recognizer.recognize(bitmap, NO_LIMIT)
        assertTrue("一个误触圆点不应被当成数字", result.digits.isEmpty())
    }

    @Test
    fun separatedDigits_yieldOneBoxEach() {
        val bitmap = blankCanvas()
        val canvas = Canvas(bitmap)
        stroke(canvas, 120f, 380f, 120f, 200f)
        stroke(canvas, 500f, 380f, 500f, 200f)
        val boxes = recognizer.segment(bitmap)
        assertEquals("两个分离的笔画应切出两个框，实际 $boxes", 2, boxes.size)
        val result = recognizer.recognize(bitmap, NO_LIMIT)
        assertEquals(2, result.digits.size)
    }

    @Test
    fun multiStrokesOfOneDigit_areNotSplitApart() {
        // "5" 的典型写法：上面一横、下面一个主体，两笔在垂直方向分离
        val bitmap = blankCanvas()
        val canvas = Canvas(bitmap)
        stroke(canvas, 150f, 180f, 260f, 180f)
        stroke(canvas, 150f, 300f, 260f, 300f)
        stroke(canvas, 150f, 300f, 150f, 420f)
        stroke(canvas, 150f, 420f, 260f, 420f)
        val boxes = recognizer.segment(bitmap)
        assertEquals("同一个数字的多笔不应被切成多个数字，实际 $boxes", 1, boxes.size)
        assertEquals(1, recognizer.recognize(bitmap, NO_LIMIT).digits.size)
    }

    @Test
    fun maxDigits_truncatesButReportsTotal() {
        val bitmap = blankCanvas()
        val canvas = Canvas(bitmap)
        for (i in 0 until 6) {
            val x = 90f + i * 150f
            stroke(canvas, x, 380f, x, 200f)
        }
        val result = recognizer.recognize(bitmap, maxDigits = 4)
        assertEquals("应报告实际切出的位数", 6, result.totalCount)
        assertEquals("识别结果应被截断到 4 位", 4, result.digits.size)
    }

    @Test
    fun recognizedDigits_areInRange() {
        val bitmap = blankCanvas()
        val canvas = Canvas(bitmap)
        stroke(canvas, 120f, 380f, 120f, 200f)
        stroke(canvas, 280f, 380f, 280f, 200f)
        stroke(canvas, 440f, 380f, 440f, 200f)
        for (digit in recognizer.recognize(bitmap, NO_LIMIT).digits) {
            assertTrue("识别的每一位都必须在 0..9，实际 $digit", digit in 0..9)
        }
    }

    private fun blankCanvas(): Bitmap =
        Bitmap.createBitmap(CANVAS_WIDTH, CANVAS_HEIGHT, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.WHITE)
        }

    /** 与 FingerPaintView 相同的笔形（圆头、圆角、抗锯齿）。 */
    private fun stroke(canvas: Canvas, fromX: Float, fromY: Float, toX: Float, toY: Float) {
        val paint = Paint().apply {
            color = Color.BLACK
            isAntiAlias = true
            isDither = true
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            strokeWidth = STROKE_WIDTH
        }
        val path = Path().apply {
            moveTo(fromX, fromY)
            lineTo(toX, toY)
        }
        canvas.drawPath(path, paint)
    }

    companion object {
        private const val CANVAS_WIDTH = 1000
        private const val CANVAS_HEIGHT = 600
        /** 与 FingerPaintView.strokeWidth 保持一致。 */
        private const val STROKE_WIDTH = 32f
        private const val NO_LIMIT = 32
    }
}
