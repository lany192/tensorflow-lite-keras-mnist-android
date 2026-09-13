import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 复现性
tf.random.set_seed(42)
np.random.seed(42)

# 0~9 是数字，10 是小数点。
#
# Android 端 MnistPreprocessor 把每个字形的**紧包围盒**缩放到最长边 20px 再按质心居中，
# 这决定了两件事，训练样本必须严格照此生成，否则模型只会学到"完美椭圆才是点"：
#   1) 点到达模型时长轴恒为 20px，短轴 = 20 / 宽高比 —— 与手指实际画了多大无关。
#      早期版本让半径在 7~10 之间随机（长轴 14~20），大多数训练点比真实输入小。
#   2) 形态只由固有宽高比、朝向、边缘软硬决定。"点"常常是手指带出的一小段短划，
#      朝向任意、可以很扁长。但切分器只看**紧包围盒**的宽高比（0.65~1.55）与填充率，
#      于是斜向拖尾（包围盒接近方形）能到达模型，固有宽高比却可以到 2.6；
#      同样扁长的水平拖尾、以及斜向的椭圆则会被切分器挡掉。
#      生成时必须复现这层筛选，否则要么漏掉真实的失败形态（旧模型在斜向拖尾上
#      召回 0.19），要么拿永远到不了模型的样本稀释点类。
DIGIT_CLASSES = 10
DECIMAL_POINT_CLASS = 10
NUM_CLASSES = DIGIT_CLASSES + 1
DOT_TRAIN_COUNT = 12_000
DOT_TEST_COUNT = 1_000

MODEL_SIZE = 28
CONTENT_SIZE = 20  # MnistPreprocessor 把紧包围盒的最长边缩放到该尺寸
INK_THRESHOLD = 32 / 255.0  # 与 InkSegmenter.inkThreshold 一致，决定包围盒边界
# InkSegmenter 判定"点候选"的几何门槛。切分器在原始像素上检查紧包围盒，而归一化是等比
# 缩放、这两个量都不变，所以同一组阈值既筛训练样本，也描述真实可达的形状。
# 关键后果：斜向拖尾的包围盒接近方形，能通过；同样扁长的水平拖尾会被拒；斜向的**椭圆**
# 也会因四个角太空（填充率 < 0.5）被拒——手指圆头笔拖出的等宽胶囊才是能到达模型的长拖尾。
DOT_ASPECT_MIN = 0.65
DOT_ASPECT_MAX = 1.55
DOT_FILL_RATIO = 0.50
GATE_ATTEMPTS = 24
CENTER = MODEL_SIZE / 2.0
# 生成时的工作分辨率：先在这里按"实际墨迹"画形状，再走一遍归一化缩放，
# 这样旋转、形状不规则带来的包围盒变化才会被如实还原成 20px 长轴。
RENDER_SIZE = 56
_RENDER_YY, _RENDER_XX = np.mgrid[0:RENDER_SIZE, 0:RENDER_SIZE].astype(np.float32)
_YY, _XX = np.mgrid[0:MODEL_SIZE, 0:MODEL_SIZE].astype(np.float32)

# MNIST 训练分布：白色笔画（值1.0）在黑色背景（0.0）上
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = (train_images / 255.0).astype('float32')[..., tf.newaxis]
test_images = (test_images / 255.0).astype('float32')[..., tf.newaxis]


def _center_by_mass(image):
    """复现 MnistPreprocessor 的质心居中：形状偏移会被归一化拉回画布中心。"""
    total = float(image.sum())
    if total <= 0.0:
        return image
    center_x = float((image * _XX).sum()) / total
    center_y = float((image * _YY).sum()) / total
    # 四周恒为背景，np.roll 的环绕部分也是 0，不会引入伪影。
    return np.roll(
        np.roll(image, int(round(CENTER - center_y)), axis=0),
        int(round(CENTER - center_x)),
        axis=1,
    )


def _render_shape(rng, aspect_range, edge_range, wobble_range):
    """在 RENDER_SIZE 网格上画出形状本身，尚未归一化。"""
    long_radius = RENDER_SIZE * 0.357
    # 偏向近圆：多数是手指点触留下的墨点，少数是随手带出的一小段短划。
    low, high = aspect_range
    aspect = low + (high - low) * rng.random() ** 1.8
    short_radius = long_radius / aspect
    angle = rng.uniform(0.0, np.pi)
    # 边缘软硬覆盖"小点被放大"与"大点被多级减半"两种相反的模糊程度。
    edge_width = rng.uniform(*edge_range)
    wobble = rng.uniform(*wobble_range)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    delta_x = _RENDER_XX - RENDER_SIZE / 2.0
    delta_y = _RENDER_YY - RENDER_SIZE / 2.0
    along = delta_x * cos_a + delta_y * sin_a
    across = -delta_x * sin_a + delta_y * cos_a
    if wobble > 0.0:
        # 低频扰动：手指压力不均，边缘不会是理想几何形状。
        along = along + wobble * long_radius * np.sin(across / short_radius * 1.7 + 1.3)
        across = across + wobble * short_radius * np.sin(along / long_radius * 2.1)

    if rng.random() < 0.5:
        # 胶囊：拖出一小段留下的短划，中段等宽、两端半圆。
        overhang = np.clip(np.abs(along) - (long_radius - short_radius), 0.0, None)
        distance = np.sqrt(overhang ** 2 + across ** 2) / short_radius
    else:
        # 超椭圆：p=2 是椭圆，p 越大越接近点触留下的圆角方头。
        power = rng.uniform(2.0, 4.0)
        distance = ((np.abs(along) / long_radius) ** power
                    + (np.abs(across) / short_radius) ** power) ** (1.0 / power)

    # 边界（灰度 0.5）精确定义在 distance == 1 处，软硬只改变过渡带宽度。
    # 若写成 (1 + edge_width - distance) / edge_width，形状会随 edge_width 一起胀大，
    # 长轴会漂到 27px，与归一化后恒定的 20px 不符。
    return np.clip((1.0 - distance) / edge_width + 0.5, 0.0, 1.0)


def _passes_segmenter_gate(image):
    """该形状能否作为点候选通过 InkSegmenter 的几何门槛。

    不满足的形状根本到不了模型：它们会被切分器划成数字或直接丢弃。拿这些样本训练
    只会稀释点类，所以生成时必须按同一组阈值拒绝采样。
    """
    mask = image > INK_THRESHOLD
    if not mask.any():
        return False
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    height = rows[-1] - rows[0] + 1
    width = cols[-1] - cols[0] + 1
    aspect = width / height
    if aspect < DOT_ASPECT_MIN or aspect > DOT_ASPECT_MAX:
        return False
    return mask.sum() / (width * height) >= DOT_FILL_RATIO


def _render_decimal_point(rng, aspect_range=(1.0, 2.6), edge_range=(0.08, 0.30),
                          wobble_range=(0.0, 0.10)):
    """渲染单个归一化后的小数点：长轴 20px、朝向任意、能通过切分器门槛。

    固有宽高比上限 2.6 来自填充率门槛：45° 的等宽胶囊到这个比例时，包围盒已达到
    π/4 填充率的下限。三个区间可放宽来做压力测试（见 diagnose_dot.py），默认值即真实分布。
    """
    for _ in range(GATE_ATTEMPTS):
        image = _render_shape(rng, aspect_range, edge_range, wobble_range)
        if _passes_segmenter_gate(image):
            return _normalize_like_android(image)
    # 参数被压到极端区间时可能连续不合格，退回必定过门槛的正圆。
    fallback = _render_shape(rng, (1.0, 1.0), edge_range, wobble_range)
    return _normalize_like_android(fallback)


def _normalize_like_android(image):
    """复现 MnistPreprocessor：紧包围盒 → 最长边缩放到 20px → 质心居中。

    不可省略。旋转后的圆角方头之类的包围盒会比短轴方向长出一截（对角撑开），
    真实归一化会把它重新压回 20px；直接输出会让长轴漂到 25px。
    """
    mask = image > INK_THRESHOLD
    if not mask.any():
        return np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    crop = image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    scale = CONTENT_SIZE / max(crop.shape)
    out_h = max(1, int(round(crop.shape[0] * scale)))
    out_w = max(1, int(round(crop.shape[1] * scale)))
    resized = tf.image.resize(
        crop[..., np.newaxis], [out_h, out_w], method='bilinear'
    ).numpy()[..., 0]
    canvas = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
    top = (MODEL_SIZE - out_h) // 2
    left = (MODEL_SIZE - out_w) // 2
    canvas[top:top + out_h, left:left + out_w] = resized
    return _center_by_mass(canvas).astype(np.float32)


def make_decimal_point_images(count, seed):
    """生成归一化后的手写小数点样本。"""
    rng = np.random.default_rng(seed)
    images = np.zeros((count, MODEL_SIZE, MODEL_SIZE, 1), dtype=np.float32)
    for i in range(count):
        images[i, :, :, 0] = _render_decimal_point(rng)
    return images


def append_decimal_point_samples(images, labels, count, seed):
    dots = make_decimal_point_images(count, seed)
    dot_labels = np.full(count, DECIMAL_POINT_CLASS, dtype=np.int64)
    return (
        np.concatenate([images, dots], axis=0),
        np.concatenate([labels, dot_labels], axis=0),
    )


train_images, train_labels = append_decimal_point_samples(
    train_images, train_labels, DOT_TRAIN_COUNT, seed=100
)
test_images, test_labels = append_decimal_point_samples(
    test_images, test_labels, DOT_TEST_COUNT, seed=200
)

# 数据增强：模拟手写输入的平移、旋转、缩放差异，只作用于训练集
augmentation = tf.keras.Sequential([
    layers.RandomTranslation(0.1, 0.1, fill_mode='constant'),
    layers.RandomRotation(0.06, fill_mode='constant'),
    layers.RandomZoom(0.15, fill_mode='constant'),
])

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .shuffle(len(train_images))
    .batch(128)
    .map(lambda x, y: (augmentation(x, training=True), y),
         num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(128)

# CNN 模型，输出 logits。最后一维从 10 改成 11：0~9 + 小数点。
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# 点类样本的形态差异比数字大，需要比原来更多的轮次才能收敛；早停 + 学习率衰减
# 避免在验证集上过拟合，同时把最好一轮的权重还回来。
model.fit(
    train_dataset,
    epochs=25,
    validation_data=test_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5
        ),
    ],
)

model.evaluate(test_dataset)

# 转换为 TensorFlow Lite（动态范围量化，减小体积，输入输出仍为 float32）
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 用 TFLite 解释器在测试集上验证转换后的模型精度，并单独报告小数点召回率。
interpreter = tf.lite.Interpreter(model_content=tflite_model)
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

BATCH = 256
correct_by_class = np.zeros(NUM_CLASSES, dtype=np.int64)
total_by_class = np.zeros(NUM_CLASSES, dtype=np.int64)
digits_as_dot = 0

for x, y in tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH):
    n = x.shape[0]
    interpreter.resize_tensor_input(input_detail['index'], [n, MODEL_SIZE, MODEL_SIZE, 1])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_detail['index'], x)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_detail['index'])
    predictions = tf.argmax(logits, axis=1).numpy()
    labels = y.numpy()

    for label, prediction in zip(labels, predictions):
        total_by_class[label] += 1
        if label == prediction:
            correct_by_class[label] += 1
        elif label != DECIMAL_POINT_CLASS and prediction == DECIMAL_POINT_CLASS:
            digits_as_dot += 1

total_correct = int(correct_by_class.sum())
total_count = int(total_by_class.sum())
dot_recall = correct_by_class[DECIMAL_POINT_CLASS] / total_by_class[DECIMAL_POINT_CLASS]

print(f"TFLite 模型测试集准确率: {total_correct / total_count:.4f}")
print(f"小数点召回率: {dot_recall:.4f} "
      f"({correct_by_class[DECIMAL_POINT_CLASS]}/{total_by_class[DECIMAL_POINT_CLASS]})")
# 反向指标同样重要：数字被判成点会在结果里凭空插进一个小数点，比漏识别更难接受。
print(f"数字被误判成小数点: {digits_as_dot}/{total_count - total_by_class[DECIMAL_POINT_CLASS]}")

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
