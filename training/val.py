# -*- CODING: UTF-8 -*-
# @time 2023/3/6 18:42
# @Author tyqqj
# @File val.py


import numpy as np
import sympy as sp
import os
import matplotlib.pyplot as plt

import cv2
# 导入hog特征提取
import model.hog as hog


def one_predict(clf, x_test):
    # 如果是一张图片
    if x_test.shape != (28, 28):
        # 报错
        print("The shape of x_test is wrong!")
        return
    # 显示图像
    cv2.imshow("x_test", x_test)
    # 转换为hog特征
    x_test = hog.hog(x_test)
    x_test = x_test.reshape(1, -1)

    # 预测
    predict = clf.predict(x_test)
    print("The predict is: ", predict)
    return predict


def val(clf, x_test, t_test):
    acc = 0
    # 处理标签
    t_test = np.argmax(t_test, axis=1)
    for i in range(len(x_test)):
        # 转换为hog特征
        x_test[i] = hog.hog(x_test[i])
        x_test[i] = x_test[i].reshape(1, -1)
        # 预测
        predict = clf.predict(x_test[i])
        if predict == t_test[i]:
            acc += 1
    acc /= len(x_test)
    print("The acc is: ", acc)
    return acc
