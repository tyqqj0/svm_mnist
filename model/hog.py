# -*- CODING: UTF-8 -*-
# @time 2023/3/4 15:17
# @Author tyqqj
# @File hog.py


import numpy as np
import sympy as sp
import os
import matplotlib.pyplot as plt

import cv2

__all__ = ["hog", "hog_feature"]

args = {"image_size": 28,  # 图像大小
        "block_size": 8,  # 块大小
        "block_stride": 4,  # 块步长
        "cell_size": 4,  # 单元格大小
        "nbins": 9,  # 梯度方向个数
        }


# 读取一张图像并返回hog特征
def hog(imgs):
    # 加载hog特征检测器
    hog = cv2.HOGDescriptor((args["image_size"], args["image_size"]), (args["block_size"], args["block_size"]),
                            (args["block_stride"], args["block_stride"]), (args["cell_size"], args["cell_size"]),
                            args["nbins"])
    # 转换为cv8u类型
    # cv2.namedWindow('img0', 8)
    # cv2.namedWindow('img1', 8)
    # cv2.imshow('img0', imgs)
    # 显示图像数据类型
    # print(imgs.dtype)
    # 转换到opencv的图像格式
    # imgs = imgs.astype(np.uint8)
    # 显示图像
    # cv2.imshow('img1', imgs)

    # 计算hog特征
    hog_feature = hog.compute(imgs)
    # for i in range(len(hog_feature)):
    # hog_feature[i] = hog_feature[i] / np.linalg.norm(hog_feature[i])
    # print(hog_feature[i])
    return hog_feature



# 将图像集合转换为hog特征
def hog_feature(imgs):
    hog_feature = []
    for i in range(len(imgs)):
        img = imgs[i].reshape(28, 28)
        hog_feature.append(hog(img))
    hog_feature = np.array(hog_feature)
    return hog_feature


# 在图像上绘制hog特征
# TODO: 画出hog特征
#     可使用skimage.feature.hog()函数


def draw_hog(hog_feature):
    # 设置窗口大小
    cv2.namedWindow('imgs', cv2.WINDOW_AUTOSIZE)
    # 生成图像
    img = np.zeros((args["image_size"], args["image_size"]), dtype=np.uint8)
    # 计算单元格个数
    cell_num = int(args["image_size"] / args["cell_size"])
    # 计算梯度方向个数
    bin_num = int(args["nbins"])
    # 计算单元格的梯度方向
    for i in range(cell_num):
        for j in range(cell_num):
            for k in range(bin_num):
                # 计算梯度方向
                angle = k * np.pi / bin_num
                # 计算梯度大小
                magnitude = hog_feature[i * cell_num + j][k]
                # 计算梯度在图像上的位置
                x = int(i * args["cell_size"] + args["cell_size"] / 2 + magnitude * np.cos(angle))
                y = int(j * args["cell_size"] + args["cell_size"] / 2 + magnitude * np.sin(angle))
                # 绘制梯度
                cv2.line(img,
                         (i * args["cell_size"] + args["cell_size"] / 2, j * args["cell_size"] + args["cell_size"] / 2),
                         (x, y), 255)
    # 显示图像
    cv2.imshow('imgs', img)
    pass
