# -*- CODING: UTF-8 -*-
# @time 2023/3/6 19:33
# @Author tyqqj
# @File draw.py


import numpy as np
# import sympy as sp
# import os
# import matplotlib.pyplot as plt


import cv2 as cv

args = {"scale": 10,
        "size": 28
        }

drawing = False  # 按下鼠标则为真
savepath = 'C:/Users/11037/Desktop/12.jpg'  # 图片保存位置
R = 0
G = 0
B = 0
img_size = args["size"] * args["scale"]
img = np.zeros((img_size, img_size, 3), np.uint8)


def nothing(x):
    pass


def draw(event, x, y, flags, param):
    global drawing, R, G, B, img
    if event == cv.EVENT_LBUTTONDOWN:  # 响应鼠标按下
        drawing = True
    elif event == cv.EVENT_MOUSEMOVE:  # 响应鼠标移动
        if drawing == True:
            img[y:y + 15, x:x + 15] = (255, 255, 255)
    elif event == cv.EVENT_LBUTTONUP:  # 响应鼠标松开
        drawing = False


def draw_fuc():
    # 创建一个黑色的图像，一个窗口
    global img
    # 清除画板
    img = np.zeros((img_size, img_size, 3), np.uint8)
    cv.namedWindow('image')
    # 创建颜色变化的轨迹栏
    save = 'OK'
    clear = 'clear'

    cv.createTrackbar(clear, 'image', 0, 1, nothing)
    cv.createTrackbar(save, 'image', 0, 1, nothing)
    cv.setMouseCallback('image', draw)
    # 将画板设为黑色
    while (1):
        # print("start draw")
        cv.imshow('image', img)
        if cv.waitKey(1) & 0xFF == 27:  # 按下esc退出
            break
        s = cv.getTrackbarPos(save, 'image')
        c = cv.getTrackbarPos(clear, 'image')
        # 返回图片
        if s == 1:
            # 将图片大小转换为28*28
            img1 = cv.resize(img, (args["size"], args["size"]))
            # 返回图片
            cv.destroyAllWindows()
            return img1
        # 清空画布
        if c == 1:
            cv.setTrackbarPos(clear, 'image', 0)
            img[:] = (0, 0, 0)


# draw_fuc()
