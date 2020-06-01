import numpy as np
import cv2
from matplotlib import pyplot as plt
img1 = cv2.imread('./DSC01739.JPG',0)
img2 = cv2.imread('./DSC01741.JPG',0)
img1 = cv2.resize(img1, dsize=(258, 258), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, dsize=(258, 258), interpolation=cv2.INTER_AREA)

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

plt.imshow(img3),plt.show()