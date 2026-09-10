# TensorFlow Keras训练Mnist示例

## 模型的训练

运行python文件夹下的keras_mnist_tflite.py（CNN模型+数据增强），直接生成tflite模型 model.tflite

## 使用模型

将训练生成的模型文件model.tflite拷贝到assets文件夹，供android读取

## 识别多个数字

在画布上从左到右依次写下多个数字，数字之间留出空隙，点击「识别」即可得到整串数字的值。
识别结果下方会列出每一位，便于区分"切分错了"还是"认错了"。

实现上模型仍只分类单个字符，多数字能力完全在 Android 端：

1. `InkSegmenter`（纯 Kotlin，不含任何 Android 依赖）把整张画布按 8 连通域切成 N 个数字的包围盒；
2. `MnistPreprocessor` 对每个包围盒单独做一次 MNIST 风格归一化；
3. `MainActivity` 逐个推理后按从左到右的顺序拼接结果。

`InkSegmenter` 的算法可用 `./gradlew test` 在 JVM 上直接验证，回归用例在
`app/src/test/kotlin/com/github/lany192/mnist/InkSegmenterTest.kt`。

注意：识别精度几乎完全由「笔画宽度 / 数字高度」的比例决定（实测安全区 ≤0.19，0.2~0.25 开始退化，0.3 以上崩塌），
而不是由切分算法或模型决定。笔宽固定为 32px，所以**数字写得大一些、位数少一些，准确率明显更高**：
位数越多，每个数字越小，比例就越容易越过悬崖。界面因此把一次识别的位数上限定为 4 位。

