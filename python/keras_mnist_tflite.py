import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 复现性
tf.random.set_seed(42)
np.random.seed(42)

# 0~9 是数字，10 是小数点。
# Android 端 MnistPreprocessor 会把每个紧包围盒缩放成约 20x20 并居中，所以小数点
# 训练样本不能按原始尺寸生成，而要模拟“实心、略椭圆的 20px 墨点”这一归一化后形态。
DIGIT_CLASSES = 10
DECIMAL_POINT_CLASS = 10
NUM_CLASSES = DIGIT_CLASSES + 1
DOT_TRAIN_COUNT = 12_000
DOT_TEST_COUNT = 1_000

# MNIST 训练分布：白色笔画（值1.0）在黑色背景（0.0）上
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = (train_images / 255.0).astype('float32')[..., tf.newaxis]
test_images = (test_images / 255.0).astype('float32')[..., tf.newaxis]


def make_decimal_point_images(count, seed):
    """生成归一化后的手写小数点样本。

    真实的点经过 bbox 裁剪和缩放后，会被放大成接近 20x20 的实心椭圆。这里随机化
    半径、宽高比、中心偏移和边缘软硬度，覆盖手指落点、笔画粗细和抗锯齿差异。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:28, 0:28].astype(np.float32)
    images = np.zeros((count, 28, 28, 1), dtype=np.float32)

    for i in range(count):
        center_x = 14.0 + rng.normal(0.0, 1.0)
        center_y = 14.0 + rng.normal(0.0, 1.0)
        radius_x = rng.uniform(7.0, 10.0)
        radius_y = radius_x * rng.uniform(0.78, 1.22)
        edge_width = rng.uniform(0.05, 0.28)

        distance = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
        image = np.clip((1.0 + edge_width - distance) / edge_width, 0.0, 1.0)
        images[i, :, :, 0] = image.astype(np.float32)

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
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.fit(train_dataset, epochs=10, validation_data=test_dataset)

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

for x, y in tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH):
    n = x.shape[0]
    interpreter.resize_tensor_input(input_detail['index'], [n, 28, 28, 1])
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

total_correct = int(correct_by_class.sum())
total_count = int(total_by_class.sum())
dot_recall = correct_by_class[DECIMAL_POINT_CLASS] / total_by_class[DECIMAL_POINT_CLASS]

print(f"TFLite 模型测试集准确率: {total_correct / total_count:.4f}")
print(f"小数点召回率: {dot_recall:.4f} "
      f"({correct_by_class[DECIMAL_POINT_CLASS]}/{total_by_class[DECIMAL_POINT_CLASS]})")

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
