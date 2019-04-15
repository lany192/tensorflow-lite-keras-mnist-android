# Android MNIST With TFLite

## Python部分

训练模型来源于TensorFlow的[basic_classification](https://www.tensorflow.org/tutorials/keras/basic_classification)示例，使用TensorFlow Keras API。

为了能够更好的在Android手机上呈现并供用户测试，训练模型里使用MNIST，而非basic classification示例里的Fashion MNIST。

本项目python源码位于根目录[python_code](https://github.com/Cyberwaif/AndroidMnistWithTFLite/tree/master/python_code)路径下。

```shell
python keras_mnist_train.py
```

**注意：** 考虑到网络问题，请自行下载MNIST数据，并配置好路径
训练时会先将图像数据数值范围从0-255转为0-1，预测时需要对待测数据做同样处理。
```python
# you can download mnist from http://yann.lecun.com/exdb/mnist/
train_images = read_local_mnist.load_train_images('input_data/train-images.idx3-ubyte')
train_labels = read_local_mnist.load_train_labels('input_data/train-labels.idx1-ubyte')
test_images = read_local_mnist.load_test_images('input_data/t10k-images.idx3-ubyte')
test_labels = read_local_mnist.load_test_labels('input_data/t10k-labels.idx1-ubyte')
```

训练得到keras_mnist_model.h5训练结果，验证h5是否有效
```shell
python eveluate.py keras_mnist_model.h5
```

将h5结果转化为tflite
```shell
python convert.py keras_mnist_model.h5
```
**注意：** 由于TensorFlow版本的持续更新，运行时可能会报`TFLiteConverter` Not Found等问题，建议使用TensorFlow Nightly，或者在[Google Colab](https://colab.research.google.com/)上进行。

## Android部分

UI逻辑来源于[MindOrks](https://github.com/MindorksOpenSource)的[AndroidTensorFlowMNISTExample](https://github.com/MindorksOpenSource/AndroidTensorFlowMNISTExample)

核心代码就是以下一小段：

```java
Interpreter mInterpreter = new Interpreter(loadModelFile(mContext));
float[][] labelProbArray = new float[1][10];
//Get input pixels from DrawView.
mInterpreter.run(userInputPixels, labelProbArray);
return getMax(labelProbArray[0]);
```