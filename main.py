# -*- CODING: UTF-8 -*-
# @time 2023/3/3 22:34
# @Author tyqqj
# @File main.py


import numpy as np
import sympy as sp
import os
import matplotlib.pyplot as plt

import cv2

from mnist_loader import load_mnist

# 读取数据
(x_train, t_train), (x_test, t_test) = load_mnist('D:\Data\mnist')
# 显示图像
img = x_train[0]
label = t_train[0]
print(label)
print(img.shape)
# 设置窗口大小
cv2.namedWindow('img', 8)
cv2.imshow('img', img)
# 等待
cv2.waitKey(0)
