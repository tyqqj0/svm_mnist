# -*- CODING: UTF-8 -*-
# @time 2023/3/3 22:34
# @Author tyqqj
# @File main.py


import numpy as np

import cv2

from mnist_loader import load_mnist
from model import hog
from sklearn import svm

import training.train as train
import training.val as val
import model.svm as svm

# 处理标签


if __name__ == '__main__':
    # 读取数据
    (x_train, t_train), (x_test, t_test) = load_mnist('D:\Data\mnist')

    # 生成模型
    model = svm.svm_model()  #

    # 训练
    clf = train.train_svm(model, x_train, t_train, train_size=1000)  # 很慢不要开太大

    # 验证
    # val.val(clf, x_test, t_test)

    # 画板测试
    c = True
    while c:
        val.draw_test(clf)
        # 等待图片显示
        # cv2.waitKey(0)
        c = input("Continue? (y/n): ")
        c = c == 'y'
        cv2.destroyAllWindows()
        # TODO:显示优化

    pass
