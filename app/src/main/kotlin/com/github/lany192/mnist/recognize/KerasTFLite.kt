package com.github.lany192.mnist.recognize

import android.content.Context
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import org.tensorflow.lite.Interpreter

class KerasTFLite(context: Context) {
    private val mInterpreter: Interpreter

    init {
        val file = loadModelFile(context)
        mInterpreter = Interpreter(file)
    }

    /**
     * 识别单个归一化字形，返回模型类别下标。
     *
     * 0~9 是数字，[DECIMAL_POINT_CLASS] 是小数点；[input] 为空或长度不为 784 时返回 -1。
     */
    fun classify(input: FloatArray?): Int {
        if (input == null || input.size != MODEL_SIZE * MODEL_SIZE) return -1
        val input4d = Array(1) { Array(MODEL_SIZE) { Array(MODEL_SIZE) { FloatArray(1) } } }
        for (i in input.indices) {
            input4d[0][i / MODEL_SIZE][i % MODEL_SIZE][0] = input[i]
        }
        val output = Array(1) { FloatArray(OUTPUT_CLASS_COUNT) }
        mInterpreter.run(input4d, output)
        return getMax(output[0])
    }

    fun run(input: FloatArray?): String? = when (val symbol = classify(input)) {
        in 0..9 -> symbol.toString()
        DECIMAL_POINT_CLASS -> "."
        else -> null
    }

    @Throws(IOException::class)
    private fun loadModelFile(context: Context): File {
        val modelFile = "model.tflite"
        val filePath = context.filesDir.path + File.separator + modelFile
        val file = File(filePath)
        // 每次都从 assets 覆盖拷贝：filesDir 在 APK 更新后仍保留旧文件，
        // 若按存在性跳过拷贝，会一直加载到旧模型
        val assetManager = context.assets
        assetManager.open(modelFile).use { stream ->
            BufferedOutputStream(FileOutputStream(file)).use { output ->
                val buffer = ByteArray(8192)
                var read: Int
                while (stream.read(buffer).also { read = it } != -1) {
                    output.write(buffer, 0, read)
                }
            }
        }
        return file
    }

    private fun getMax(results: FloatArray): Int {
        var maxID = 0
        var maxValue = results[maxID]
        for (i in 1..<results.size) {
            if (results[i] > maxValue) {
                maxID = i
                maxValue = results[i]
            }
        }
        return maxID
    }

    fun release() {
        mInterpreter.close()
    }

    companion object {
        const val DECIMAL_POINT_CLASS = 10
        private const val OUTPUT_CLASS_COUNT = 11
        private const val MODEL_SIZE = 28
    }
}
