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
from sklearn import svm


def train_svm(x_train, t_train):
    # 训练
    clf = svm.SVC()
    clf.fit(x_train, t_train)
    return clf


def one_predict(clf, x_test):
    # 如果是一张图片
    if x_test.shape != (28, 28):
        # 报错
        print("The shape of x_test is wrong!")
        return
    # 转换为hog特征
    x_test = hog.hog(x_test)
    x_test = x_test.reshape(1, -1)

    # 预测
    predict = clf.predict(x_test)
    print("The predict is: ", predict)
    return predict


# 处理标签
def label(t_train):
    # 处理标签
    t_train = np.array(t_train)
    label = []
    for i in range(len(t_train)):
        for j in range(10):
            if t_train[i][j] == 1:
                label.append(j)
    label = np.array(label)
    return label


if __name__ == '__main__':
    # 读取数据
    (x_train, t_train), (x_test, t_test) = load_mnist('D:\Data\mnist')
    # 显示图像
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', x_test[0].reshape(28, 28))
    # 将所有图片提取hog特征
    hog_feature = hog.hog_feature(x_train)

    # 处理标签
    t_train = label(t_train)

    # 训练
    # print("here")
    clf = train_svm(hog_feature[:1000], t_train[:1000])
    # print("here")
    # 显示
    one_predict(clf, x_test[0])
    # print("here")
    cv2.waitKey(0)
    pass
