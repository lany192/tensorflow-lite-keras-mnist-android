package com.github.lany192.mnist

import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.github.lany192.mnist.databinding.ActivityMainBinding

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
        val PIXEL_SIZE = 28
        val bitmap = binding.fingerPaintView.exportToBitmap(PIXEL_SIZE, PIXEL_SIZE)
        val pixels = getPixelData(bitmap)
        //should be same format with train
        for (i in pixels.indices) {
            pixels[i] = pixels[i] / 255
        }
        for (i in 0..<PIXEL_SIZE) {
            val a = pixels.copyOfRange(i * PIXEL_SIZE, i * PIXEL_SIZE + PIXEL_SIZE)
            Log.v("test", a.contentToString())
        }
        binding.textResult.text = "数字是: ${mTFLite?.run(pixels)}"
    }

    private fun onClearClicked() {
        binding.fingerPaintView.clear()
        binding.textResult.text = ""
    }

    /**
     * Get 28x28 pixel data for tensorflow input.
     */
    fun getPixelData(bitmap: Bitmap?): FloatArray {
        if (bitmap == null) {
            return FloatArray(0)
        }
        val width = bitmap.getWidth()
        val height = bitmap.getHeight()
        // Get 28x28 pixel data from bitmap
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val retPixels = FloatArray(pixels.size)
        for (i in pixels.indices) {
            // Set 0 for white and 255 for black pixel
            val pix = pixels[i]
            val b = pix and 0xff
            retPixels[i] = (0xff - b).toFloat()
        }
        return retPixels
    }
}

