#!/usr/bin/python3

from models import *
from data_support import load_mnist_data, load_num_image
from keras.models import load_model
import numpy as np
import cv2


def main():
    num = input('Enter a number range 0-9: ')
    if int(num) > 10 or int(num) < 0:
        print("the number not range 0-9")
        return
    rec_train()
    rec_test(num)


def rec_train():
    # 1、模型构建
    # @see https://keras.io/zh/getting-started/sequential-model-guide/
    model = build_simple_fnn_model(784, 512, 10)

    # 2、模型编译
    # @see https://keras.io/zh/models/sequential/
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 3、模型训练
    print("load mnist data")
    x_train, x_test, y_train, y_test = load_mnist_data()
    print("fit data")
    model.fit(
        x=x_train,
        y=y_train,
        epochs=1,
        batch_size=1000,
        verbose=1,
        initial_epoch=0)

    # 4、测试验证
    print("evaluate data")
    scores = model.evaluate(x_test, y_test, batch_size=1000, verbose=1)
    print("scores=%s" % scores)

    # 5、保存模型
    # @see https://keras.io/zh/visualization/
    print("save model")
    model.save("./number_model.h5")
    del model


def rec_test(num):
    image_data = load_num_image(num)
    model = load_model("./number_model.h5")
    score = model.predict(image_data)

    max_val_index = np.argmax(score[0])
    max_confidence = score[0][max_val_index]
    # 输出识别结果
    print("score:", score)
    print("max_val_index:", max_val_index)
    print("max_confidence:", max_confidence)
    # 展示识别结果展示
    image_path = './image/number/' + str(max_val_index + 1) + '.jpg'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("recNumber", img)
    cv2.waitKey(4000)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
