import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  # 因为在数据增强的时候需要4列，所以需要扩充一列，单列是灰度值

image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,
    rotation_range=30,  # 随机偏转30度
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=False,
    zoom_range=0.5
)
image_gen_train.fit(x_train)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------正在读取模型-----------------')
    model.load_weights(checkpoint_save_path)  # 读取路径

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)  # 保存最优模型

history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=20, validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback])
model.summary()


