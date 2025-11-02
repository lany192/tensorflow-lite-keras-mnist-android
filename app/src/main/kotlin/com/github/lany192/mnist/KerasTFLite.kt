package com.github.lany192.mnist

import android.content.Context
import android.os.Environment
import org.tensorflow.lite.Interpreter
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream

class KerasTFLite(context: Context) {
    private val mInterpreter: Interpreter

    init {
        val file = loadModelFile(context)
        mInterpreter = Interpreter(file)
    }

    fun run(input: FloatArray?): String? {
        //result will be number between 0~9
        val labelProbArray = Array<FloatArray?>(1) { FloatArray(10) }
        mInterpreter.run(input, labelProbArray)
        val labels: MutableList<String?> = ArrayList()
        for (i in 0..9) {
            labels.add(i.toString())
        }
        return labels[getMax(labelProbArray[0]!!)]
    }

    @Throws(IOException::class)
    private fun loadModelFile(context: Context): File {
        val modelFile = "keras_mnist_model.tflite"
        val filePath = context.filesDir.path + File.separator + modelFile
        val file = File(filePath)
        if (!file.exists()) {
            val assetManager = context.assets
            val stream = assetManager.open(modelFile)
            val output: OutputStream = BufferedOutputStream(FileOutputStream(filePath))
            val buffer = ByteArray(1024)
            var read: Int
            while ((stream.read(buffer).also { read = it }) != -1) {
                output.write(buffer, 0, read)
            }
            stream.close()
            output.close()
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
}
