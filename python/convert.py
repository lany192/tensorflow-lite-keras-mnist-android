import argparse
import sys

import tensorflow as tf


def convert2tflite(modelname):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(modelname)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Please input model file path as argv")
        quit(0)

    convert2tflite(sys.argv[1])

