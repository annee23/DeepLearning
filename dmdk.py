import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from scipy import misc
from PIL import Image
img_dir = "./img"
data_path = img_dir + "/*.JPG"
files = glob.glob(data_path)
data = []

for i,item in enumerate(files):
    img = cv2.imread(files[i])
    img2 = cv2.imread(files[i+1])
    img = cv2.resize(img, dsize=(500,500), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, dsize=(500,500), interpolation= cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img), plt.show()