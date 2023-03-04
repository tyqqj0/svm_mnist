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
import hog

if __name__ == '__main__':
    # 读取数据
    (x_train, t_train), (x_test, t_test) = load_mnist('D:\Data\mnist')
    # 显示图像
    img = x_train[0]
    label = t_train[0]
    print(label)
    print(img.shape)
    # 设置窗口大小
    cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('img', img)
    # 加载hog特征检测器
    # hog = cv2.HOGDescriptor((28, 28), (8, 8), (4, 4), (4, 4), 9)
    # 转换为cv8u类型
    # img = img.astype(np.uint8)
    print("here")
    # 计算hog特征
    # hog_feature = hog.compute(img)
    hog_feature = hog.hog(img)
    # hog.draw_hog(hog_feature)
    print(hog_feature.shape)
    # 显示图像
    # cv2.imshow('img', hog_feature)
    # 等待
    cv2.waitKey(0)
    pass
