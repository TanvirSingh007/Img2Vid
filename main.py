# Image Slide Show using addWeighted

import cv2
import numpy as np
from math import ceil
import os
from PIL import Image


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# Getting images from image base
dst = "./MyFolder/"
images = os.listdir(dst)
length = len(images)
bg = cv2.imread("./BG/sky.png")
images.sort()
result = np.zeros((900, 1600, 3), np.uint8)
i = 0
imgArray = []

a = 1.0  # alpha
b = 0.0  # beta
img = cv2.imread(dst + images[i])

img = image_resize(img, height=900)
height, width, layers = img.shape
x_dist = int((1600 - width) / 2)
bg_coverted = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
pil_bg = Image.fromarray(bg_coverted)

color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(color_coverted)

pil_image.thumbnail((1600, 900))
Image.Image.paste(pil_bg, pil_image, (x_dist, 0))
# print(pil_image.size)

numpy_image = np.array(pil_bg)
img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
img = cv2.resize(img, (1600, 900))

# Slide Show Loop
while (True):

    if (ceil(a) == 0):
        a = 1.0
        b = 0.0
        i = (i + 1) % length  # Getting new image from directory
        img = cv2.imread(dst + images[i])
        img = image_resize(img, height=900)
        height, width, layers = img.shape

        bg_coverted = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        pil_bg = Image.fromarray(bg_coverted)

        x_dist = int((1600 - width) / 2)

        color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)

        pil_image.thumbnail((1600, 900))

        Image.Image.paste(pil_bg, pil_image, (x_dist, 0))

        numpy_image = np.array(pil_bg)
        img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    a -= 0.01
    b += 0.01

    # Image Transition from one to another
    result = cv2.addWeighted(result, a, img, b, 0)
    imgArray.append(result)
    key = cv2.waitKey(10) & 0xff
    if i == 7 and b > 1:
        break

# print(len(imgArray))

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1600, 900))

for i in range(len(imgArray)):
    out.write(imgArray[i])
    # print(img_array[i])
out.release()
