package com.github.lany192.mnist

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs

class FingerPaintView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    private var path: Path? = null
    private var drawingBitmap: Bitmap? = null
    private var drawingCanvas: Canvas? = null
    private var drawingPaint: Paint? = null
    private var penX = 0.0f
    private var penY = 0.0f
    private var paint: Paint? = null

    var isEmpty: Boolean = true
        private set

    init {
        drawingPaint = Paint(Paint.DITHER_FLAG)
        path = Path()
        paint = Paint()
        paint!!.isAntiAlias = true
        paint!!.isDither = true
        paint!!.setColor(Color.BLACK)
        paint!!.style = Paint.Style.STROKE
        paint!!.strokeCap = Paint.Cap.ROUND
        paint!!.strokeJoin = Paint.Join.ROUND
        // MNIST 模型对"笔画宽 / 数字高度"的比例极敏感：实测安全区在 0.19 以下，
        // 0.2~0.25 开始明显退化、0.3 以上崩塌。64f 只在把字写成占满整个视图（≈700px 高）
        // 时才是最优；一旦写多位数、每个数字变小，64f 的比例会飙到 0.45 以上，整串全对率归零。
        // 32f 让安全区覆盖约 170~800px 的字高，且大字单数字场景实测无损失（97.5% vs 96.7%）。
        paint!!.strokeWidth = 32f
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        drawingBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        drawingCanvas = Canvas(drawingBitmap!!)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawBitmap(drawingBitmap!!, 0f, 0f, drawingPaint)
        canvas.drawPath(path!!, paint!!)
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {
        if (event == null) return false
        isEmpty = false
        val x = event.x
        val y = event.y
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                path!!.reset()
                path!!.moveTo(x, y)
                penX = x
                penY = y
                invalidate()
            }

            MotionEvent.ACTION_MOVE -> {
                val dx = abs(x - penX)
                val dy = abs(y - penY)
                val touchTolerance = 4f
                if (dx >= touchTolerance || dy >= touchTolerance) {
                    path!!.quadTo(penX, penY, (x + penX) / 2, (y + penY) / 2)
                    penX = x
                    penY = y
                }
                invalidate()
            }

            MotionEvent.ACTION_UP -> {
                path!!.lineTo(penX, penY)
                drawingCanvas!!.drawPath(path!!, paint!!)
                path!!.reset()
                performClick()
                invalidate()
            }
        }
        super.onTouchEvent(event)
        return true
    }

    fun clear() {
        path!!.reset()
        drawingBitmap = Bitmap.createBitmap(
            drawingBitmap!!.getWidth(),
            drawingBitmap!!.getHeight(),
            Bitmap.Config.ARGB_8888
        )
        drawingCanvas = Canvas(drawingBitmap!!)
        isEmpty = true
        invalidate()
    }

    /**
     * 导出笔画内容用于识别：纯白背景、原始分辨率。
     * 不能带入视图背景色，也不能缩放，否则预处理（裁剪/宽高比）失效。
     */
    fun exportDrawingBitmap(): Bitmap {
        val rawBitmap = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888)
        val canvas = Canvas(rawBitmap)
        canvas.drawColor(Color.WHITE)
        draw(canvas)
        return rawBitmap
    }
}
