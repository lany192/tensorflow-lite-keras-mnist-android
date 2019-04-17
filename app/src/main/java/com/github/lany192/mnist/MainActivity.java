package com.github.lany192.mnist;

import android.Manifest;
import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.nex3z.fingerpaintview.FingerPaintView;
import com.tbruyelle.rxpermissions2.RxPermissions;

import java.util.Arrays;
import java.util.concurrent.Executors;

import io.reactivex.functions.Consumer;

public class MainActivity extends AppCompatActivity {
    private final String TAG = getClass().getSimpleName();
    private TextView mResultText;
    private FingerPaintView fingerPaintView;
    private View detectButton;
    private KerasTFLite mTFLite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        fingerPaintView = findViewById(R.id.finger_paint_view);
        detectButton = findViewById(R.id.buttonDetect);
        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onDetectClicked();
            }
        });
        findViewById(R.id.buttonClear).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onClearClicked();
            }
        });
        mResultText = findViewById(R.id.textResult);
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                detectButton.setVisibility(View.VISIBLE);
            }
        });
    }

    @SuppressLint("CheckResult")
    @Override
    protected void onResume() {
        super.onResume();
        new RxPermissions(this)
                .request(Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE)
                .subscribe(new Consumer<Boolean>() {

                    @Override
                    public void accept(Boolean granted) throws Exception {
                        if (granted) {
                            Executors.newSingleThreadExecutor().execute(new Runnable() {
                                @Override
                                public void run() {
                                    try {
                                        if (mTFLite == null)
                                            mTFLite = new KerasTFLite(MainActivity.this);
                                        makeButtonVisible();
                                    } catch (final Exception e) {
                                        throw new RuntimeException("Error initializing TensorFlow!", e);
                                    }
                                }
                            });
                        } else {
                            Toast.makeText(MainActivity.this, "请授予存储权限", Toast.LENGTH_SHORT).show();
                        }
                    }
                });
    }

    @Override
    protected void onPause() {
        if (mTFLite != null) {
            mTFLite.release();
            mTFLite = null;
        }
        super.onPause();
    }

    private void onDetectClicked() {
        if (fingerPaintView.isEmpty()) {
            Toast.makeText(this, "请写上一个数字", Toast.LENGTH_SHORT).show();
            return;
        }
        final int PIXEL_SIZE = 28;
        Bitmap bitmap = fingerPaintView.exportToBitmap(PIXEL_SIZE, PIXEL_SIZE);
        float pixels[] = getPixelData(bitmap);
        //should be same format with train
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] = pixels[i] / 255;
        }
        for (int i = 0; i < PIXEL_SIZE; i++) {
            float[] a = Arrays.copyOfRange(pixels, i * PIXEL_SIZE, i * PIXEL_SIZE + PIXEL_SIZE);
            Log.v(TAG, Arrays.toString(a));
        }
        String result = mTFLite.run(pixels);
        String value = "数字是: " + result;
        mResultText.setText(value);
    }

    private void onClearClicked() {
        fingerPaintView.clear();
        mResultText.setText("");
    }

    /**
     * Get 28x28 pixel data for tensorflow input.
     */
    public float[] getPixelData(Bitmap bitmap) {
        if (bitmap == null) {
            return null;
        }
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        // Get 28x28 pixel data from bitmap
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        float[] retPixels = new float[pixels.length];
        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixel
            int pix = pixels[i];
            int b = pix & 0xff;
            retPixels[i] = 0xff - b;
        }
        return retPixels;
    }
}

