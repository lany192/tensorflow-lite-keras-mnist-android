import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 复现性
tf.random.set_seed(42)

# 加载数据集，MNIST 训练分布：白色笔画（值1.0）在黑色背景（0.0）上
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = (train_images / 255.0).astype('float32')[..., tf.newaxis]
test_images = (test_images / 255.0).astype('float32')[..., tf.newaxis]

# 数据增强：模拟手写输入的平移、旋转、缩放差异，只作用于训练集
augmentation = tf.keras.Sequential([
    layers.RandomTranslation(0.1, 0.1, fill_mode='constant'),
    layers.RandomRotation(0.06, fill_mode='constant'),
    layers.RandomZoom(0.15, fill_mode='constant'),
])

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .shuffle(60000)
    .batch(128)
    .map(lambda x, y: (augmentation(x, training=True), y),
         num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(128)

# CNN 模型，输出 logits
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(10),
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

# 用 TFLite 解释器在测试集上验证转换后的模型精度
interpreter = tf.lite.Interpreter(model_content=tflite_model)
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

BATCH = 256
interpreter.resize_tensor_input(input_detail['index'], [BATCH, 28, 28, 1])
interpreter.allocate_tensors()

correct = 0
total = 0
for x, y in tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH):
    n = x.shape[0]
    if n != BATCH:
        interpreter.resize_tensor_input(input_detail['index'], [n, 28, 28, 1])
        interpreter.allocate_tensors()
    interpreter.set_tensor(input_detail['index'], x)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_detail['index'])
    preds = tf.argmax(logits, axis=1).numpy()
    correct += (preds == y.numpy()).sum()
    total += n

print(f"TFLite 模型测试集准确率: {correct / total:.4f}")

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
