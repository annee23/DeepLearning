import os

import numpy as np

import cv2

from matplotlib import pyplot as plt

import glob

from PIL import Image

img_dir = "./img"

data_path = img_dir + "/*.JPG"

files = glob.glob(data_path)

data = []

print(files)

for f1 in files:
    img = cv2.imread(f1)

    img = cv2.resize(img, dsize=(258, 258), interpolation=cv2.INTER_AREA)

    sift = cv2.xfeatures2d.SIFT_create()

    kp = sift.detect(img, None)

    img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('img', img)

    cv2.waitKey()

    cv2.destroyAllWindows()