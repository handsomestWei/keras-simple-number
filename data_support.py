
import cv2
import numpy as np
from keras.datasets import mnist


def load_mnist_data():

    # 训练数据60000幅28*28=784，测试数据10000幅28*28=784的灰度图片
    (x_train, y_train), (x_test, y_test) = mnist.load_data("./mnist.npz")
    # 每个28*28转成一维的数据，一个label对应一个原来的60000 * 28*28转换成60000 * 784
    x_train = x_train.reshape(
        x_train.shape[0],
        x_train.shape[1] *
        x_train.shape[2])
    # 10000 * 28*28转换成10000 * 784
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    # 将index转换成一个one_hot矩阵
    y_train = (np.arange(10) == y_train[:, None]).astype(int)
    y_test = (np.arange(10) == y_test[:, None]).astype(int)

    return x_train, x_test, y_train, y_test


def load_num_image(num):
    image_path = './image/number/' + num + '.jpg'
    # 读取灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", img)
    img = cv2.resize(img, (28, 28))
    image_data_source = np.array(img)
    image_data = []
    image_data.append(
        # 每个28*28转成一维的数据，一个label对应一个原来的60000 * 28*28转换成60000 * 784
        list(np.reshape(image_data_source, 28 * 28)))
    image_data = np.array(image_data)
    return image_data
