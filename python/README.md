训练脚本会在 MNIST 的 0~9 之外合成小数点样本，生成输出维度为 11 的 `model.tflite`，
并分别打印混合测试集准确率、小数点召回率，以及数字被误判成小数点的数量。

小数点样本的形态由 Android 侧的两段处理共同决定，生成器复现了这两层：

1. `InkSegmenter` 判定"点候选"的几何门槛——紧包围盒宽高比 0.65~1.55、填充率 ≥ 0.5。
   不满足的形状根本到不了模型，所以生成时按同一组阈值拒绝采样。
2. `MnistPreprocessor` 的归一化——包围盒最长边缩放到 20px 再质心居中，因此点到达
   模型时长轴恒为 20px，形态只由固有长短轴比、朝向和边缘软硬决定。

关键后果是**斜向拖尾**：它的包围盒接近方形能通过第 1 步，归一化后的固有长短轴比却
可以到 2.5（斜向的椭圆则因四个角太空、填充率不足而被挡掉）。改这里的生成参数前，
先确认 Android 侧这两段行为没有变，否则模型会重新退化到"只认正圆"。

环境准备：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
pip freeze > requirements.txt
```

训练并同步到 Android assets：

```bash
python keras_mnist_tflite.py
cp model.tflite ../app/src/main/assets/model.tflite
```
