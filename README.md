# keras-simple-number
使用keras框架实现的一个简单0到9数字识别。使用mnist数据集进行训练和验证，使用cv2读取数字图片。

# Usage

## 数据集准备
windows环境下，keras读取数据集的默认目录为：
```
C:\Users\xxxx\.keras\datasets\
```
数据集mnist.npz放置在该目录下

## 数字图片准备
0-9数字图片放置在项目的`image\number\`目录下
```
{$projectPath}\image\number\
```

## 训练模型存储
训练好的模型存储在项目根目录下
```
{$projectPath}\number_model.h5
```

## 运行
```
python keras_res_num.py
```
将展示输入图片，和识别出的数字
