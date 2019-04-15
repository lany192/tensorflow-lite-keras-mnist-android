package com.github.lany192.mnist;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Environment;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class KerasTFLite {
    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";

    private static final String MODEL_FILE = "keras_mnist_model.tflite";
    private static final String LABEL_FILE = "graph_label_strings.txt";
    private static final String TAG = "KerasMNIST";
    private final Context mContext;
    private List<String> mLables;
    private Interpreter mInterpreter;
    private float[][] labelProbArray = null;

    public KerasTFLite(Context context) throws IOException {
        mContext = context;
        MappedByteBuffer byteBuffer = loadModelFile(mContext);
        mInterpreter = new Interpreter(byteBuffer);
        //result will be number between 0~9
        labelProbArray = new float[1][10];
        mLables = loadLabelList(mContext);
    }

    public String run(float[] input) {
        mInterpreter.run(input, labelProbArray);
        Log.v(TAG, Arrays.toString(labelProbArray[0]));
        return mLables.get(getMax(labelProbArray[0]));
    }

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        String filePath = Environment.getExternalStorageDirectory() + File.separator + MODEL_FILE;
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
        FileInputStream inputStream = new FileInputStream(filePath);
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    }

    private List<String> loadLabelList(Context context) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(context.getAssets().open(LABEL_FILE)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
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
