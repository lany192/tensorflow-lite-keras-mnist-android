训练脚本会在 MNIST 的 0~9 之外合成小数点样本，生成输出维度为 11 的 `model.tflite`，
并分别打印混合测试集准确率和小数点召回率。

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
