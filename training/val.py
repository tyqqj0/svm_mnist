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
import training.draw as draw






def one_predict(clf, x_test):
    img = x_test
    # 如果是一张图片
    if x_test.shape != (28, 28):
        # 报错
        print("The shape of x_test is wrong! It is: ", x_test.shape)
        return
    # 设置窗口大小
    # cv2.namedWindow("x_test", cv2.WINDOW_NORMAL)

    # 转换为hog特征
    x_test = hog.hog(x_test)
    x_test = x_test.reshape(1, -1)

    # 预测
    predict = clf.predict(x_test)
    # 打印预测结果 红色字体打印
    # 显示图像
    # 放大图像
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    img = cv2.resize(img, (28 * 10, 28 * 10))
    # cv2.resizeWindow("result", 28 * 10, 28 * 10)
    print("\033[1;31mThe predict is: \033[0m", predict)
    # 在窗口中显示预测结果
    cv2.putText(img, str(predict), (0, 28 * 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.imshow("result", img)
    # 等待照片显示 延时零点一秒
    cv2.waitKey(10)
    return predict


def val(clf, x_test, t_test):
    print("start val")
    acc = 0
    # 处理标签
    x, t = hog.make_dataset(x_test, t_test)
    for i in range(len(x)):
        # 预测
        predict = clf.predict(x[i].reshape(1, -1))
        if predict == t[i]:
            acc += 1
    acc /= len(x)
    print("The acc is: ", acc)
    return acc


# 画板功能
def draw_test(clf):
    img = draw.draw_fuc()
    # 灰度化图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("The shape of img is: ", img.shape)
    one_predict(clf, img)

    return img
