# TensorFlow and tf.keras
import os
import struct

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime

import read_local_mnist


epochs_time = 100


def train():

    # you can download mnist from http://yann.lecun.com/exdb/mnist/
    train_images = read_local_mnist.load_train_images('input_data/train-images.idx3-ubyte')
    train_labels = read_local_mnist.load_train_labels('input_data/train-labels.idx1-ubyte')
    test_images = read_local_mnist.load_test_images('input_data/t10k-images.idx3-ubyte')
    test_labels = read_local_mnist.load_test_labels('input_data/t10k-labels.idx1-ubyte')

    class_names = ['0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print(train_images[0])

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=epochs_time)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    modelname = "keras_mnist_model" + str(datetime.datetime.now().strftime('%m%d%H%M%S')) + \
                "_epochs" + str(epochs_time) + ".h5"
    model.save(modelname)

    predictions = model.predict(test_images)

    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(test_labels[0])

    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        true_label = int(true_label)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)
        print("prediction array:", predictions_array)
        print("true_label:", true_label)
        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)

    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        true_label = int(true_label)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions, test_labels)
    plt.show()

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

    # Grab an image from the test dataset
    img = test_images[0]
    print(img.shape)

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))
    print(img.shape)

    predictions_single = model.predict(img)
    print(predictions_single)

    plot_value_array(0, predictions_single, test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()



def showimage():
    train_images, train_labels = load_mnist('input_data', 'train')

    print(train_images[0])

    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    fig, ax = plt.subplots(
        nrows=5,
        ncols=5,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    for i in range(25):
        img = train_images[train_labels == 6][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True, )
    ax = ax.flatten()
    for i in range(10):
        img = train_images[train_labels == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # showimage()
    train()
