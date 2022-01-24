# Image Slide Show using addWeighted

import cv2
import numpy as np
from math import ceil
import os
import glob
import cvzone
from matplotlib import pyplot as plt

images = glob.glob('./MyFolder/*.png')
images.sort()
bg = cv2.imread('./BG/sky.png')

img_array = []
for filename in images:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (1600, 900)
    # sizeChange =
    # print(img.shape)
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # bg[0:900, 0:1600] = resized

    # combined = cvzone.overlayPNG(bg, resized, [20, 20])
    #
    # row1, cols1, ch1 = bg.shape
    # row2, cols2, ch2 = img.shape
    #
    # res = cv2.resize(img, None, fx=(1. * row1 / row2), fy=(1. * cols1 / cols2), interpolation=cv2.INTER_CUBIC)

    img_array.append(resized)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])
    # print(img_array[i])
out.release()
