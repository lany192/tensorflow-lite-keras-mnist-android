import sys

import tensorflow.keras as keras
import read_local_mnist


def eva(modelname, test_images, test_labels):
    new_model = keras.models.load_model(modelname)
    new_model.summary()
    test_images = test_images / 255.0
    loss, acc = new_model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please input model file path as argv")
        quit(0)

    test_images = read_local_mnist.load_test_images('input_data/t10k-images.idx3-ubyte')
    test_labels = read_local_mnist.load_test_labels('input_data/t10k-labels.idx1-ubyte')
    eva(sys.argv[1], test_images, test_labels)
