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
        paint!!.strokeWidth = 36f
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

    fun exportToBitmap(width: Int, height: Int): Bitmap {
        val rawBitmap = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888)
        val canvas = Canvas(rawBitmap)
        val bgDrawable = background
        if (bgDrawable != null) {
            bgDrawable.draw(canvas)
        } else {
            canvas.drawColor(Color.WHITE)
        }
        draw(canvas)
        val scaledBitmap = Bitmap.createScaledBitmap(rawBitmap, width, height, false)
        rawBitmap.recycle()
        return scaledBitmap
    }
}
