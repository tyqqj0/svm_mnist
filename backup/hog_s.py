# -*- CODING: UTF-8 -*-
# @time 2023/3/4 23:00
# @Author tyqqj
# @File hog_s.py


import numpy as np
import sympy as sp
import os
import matplotlib.pyplot as plt

import cv2
from skimage import feature, exposure

# import cv2


image = cv2.imread('image/soccer2.bmp')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fd, hog_image = feature.hog(image, orientations=3, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',
                            feature_vector=True, visualize=True)
cv2.imshow('hog', hog_image)
cv2.waitKey(0)
