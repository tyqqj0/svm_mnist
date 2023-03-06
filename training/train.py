# -*- CODING: UTF-8 -*-
# @time 2023/3/6 18:41
# @Author tyqqj
# @File train.py


import numpy as np

# import cv2
# 导入hog特征提取
import model.hog as hog

__all__ = ['train_svm']


# 训练
def train_svm(model, x_train, t_train, train_size=1000):
    # print("The shape of x_train is: ", x_train.shape)
    # print("The shape of t_train is: ", t_train.shape)
    hog_feature = hog.hog_feature(x_train)
    # 处理标签
    t_train = label(t_train)
    # print("The shape of x_train is: ", hog_feature.shape)
    # print("The shape of t_train is: ", t_train.shape)
    # 训练
    if train_size > len(x_train):
        train_size = len(x_train)
    clf = model.fit(x_train[:train_size], t_train[:train_size])
    return clf


def label(t_train):
    # 处理标签
    t_train = np.argmax(t_train, axis=1)
    return t_train
