package com.github.lany192.mnist;

import android.content.Context;
import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class KerasTFLite {
    private static final String MODEL_FILE = "model.tflite";
    private Interpreter mInterpreter;

    public KerasTFLite(Context context) throws IOException {
        File file = loadModelFile(context);
        mInterpreter = new Interpreter(file);
    }

    public String run(float[] input) {
        //result will be number between 0~9
        float[][] labelProbArray = new float[1][10];
        mInterpreter.run(input, labelProbArray);
        List<String> labels = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            labels.add(String.valueOf(i));
        }
        return labels.get(getMax(labelProbArray[0]));
    }

    private File loadModelFile(Context context) throws IOException {
        String filePath = context.getFilesDir() + File.separator + MODEL_FILE;
        File file = new File(filePath);
        if (!file.exists()) {
            AssetManager assetManager = context.getAssets();
            InputStream stream = assetManager.open(MODEL_FILE);
            OutputStream output = new BufferedOutputStream(new FileOutputStream(filePath));
            byte[] buffer = new byte[1024];
            int read;
            while ((read = stream.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            stream.close();
            output.close();
        }
        return file;
    }

    private int getMax(float[] results) {
        int maxID = 0;
        float maxValue = results[maxID];
        for (int i = 1; i < results.length; i++) {
            if (results[i] > maxValue) {
                maxID = i;
                maxValue = results[maxID];
            }
        }
        return maxID;
    }

    public void release() {
        mInterpreter.close();
    }
}
