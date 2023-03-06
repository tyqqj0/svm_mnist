# -*- CODING: UTF-8 -*-
# @time 2023/3/3 22:45
# @Author tyqqj
# @File loader.py


import numpy as np
import sympy as sp
import os
import matplotlib.pyplot as plt


# import cv2

# 读取idx文件
def load_data(path):
    with open(path, 'rb') as f:
        data = f.read()
    # 检验magic number是否全为0
    magic_number = int.from_bytes(data[:4], byteorder='big')
    if magic_number != 0:
        raise ValueError('Invalid magic number %d in MNIST file %s' % (magic_number, path))

    # 0x08: unsigned byte
    # 0x09: signed byte
    # 0x0B: short(2 bytes)
    # 0x0C: int(4 bytes)
    # 0x0D: float(4 bytes)
    # 0x0E: double(8 bytes)
    # 检验数据类型
    dtypes = int.from_bytes(data[4:8], byteorder='big')
    if dtypes == 0x08:
        data_type = np.uint8
    elif dtypes == 0x09:
        data_type = np.int8
    elif dtypes == 0x0B:
        data_type = np.short
    elif dtypes == 0x0C:
        data_type = np.int32
    elif dtypes == 0x0D:
        data_type = np.float32
    elif dtypes == 0x0E:
        data_type = np.float64
    else:
        raise ValueError('Invalid data type %d in MNIST file %s' % (dtypes, path))

    # 检验维度
    dim = int.from_bytes(data[8:12], byteorder='big')
    if dim == 1:
        # 一维数据
        length = int.from_bytes(data[12:16], byteorder='big')
        return np.frombuffer(data, dtype=data_type, offset=16, count=length)
    else:
        # 多维数据
        shape = []
        for i in range(dim):
            shape.append(int.from_bytes(data[12 + 4 * i:16 + 4 * i], byteorder='big'))
        return np.frombuffer(data, dtype=data_type, offset=16 + 4 * dim).reshape(shape, 28 * 28)

    # data_type = int.from_bytes(data[4:8], byteorder='big')
