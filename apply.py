from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model_save_path = './checkpoint/mnist.ckpt'
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 图片有10各类别，因此输出层有是个神经元
])

model.load_weights(model_save_path)
preNum = int(input("输入需要识别的图片数量:"))

for i in range(preNum):
    image_path = input("请输入要识别的图片名:")
    img = Image.open(image_path)

    image = plt.imread(image_path)
    plt.set_cmap('gray')
    plt.imshow(image)

    img = img.resize((28, 28), Image.ANTIALIAS)  # 改变图片尺寸
    img_arr = np.array(img.convert('L'))  # 转化成灰度图

    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 210:  # 灰度值《160的颜色变成纯白
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0  # 其他的变成纯黑

    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)

    print('\n')
    tf.print(pred)

    plt.pause(1)
    plt.close()
