package com.github.lany192.mnist

import android.content.Context
import android.os.Environment
import org.tensorflow.lite.Interpreter
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class KerasTFLite(context: Context) {
    private val mInterpreter: Interpreter

    init {
        val file = loadModelFile(context)
        mInterpreter = Interpreter(file)
    }

    fun run(input: FloatArray?): String? {
        if (input == null || input.size != MODEL_SIZE * MODEL_SIZE) return null
        val input4d = Array(1) { Array(MODEL_SIZE) { Array(MODEL_SIZE) { FloatArray(1) } } }
        for (i in input.indices) {
            input4d[0][i / MODEL_SIZE][i % MODEL_SIZE][0] = input[i]
        }
        val output = Array(1) { FloatArray(10) }
        mInterpreter.run(input4d, output)
        return getMax(output[0]).toString()
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
                maxValue = results[maxID]
            }
        }
        return maxID
    }

    fun release() {
        mInterpreter.close()
    }

    companion object {
        private const val MODEL_SIZE = 28
    }
}
