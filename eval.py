import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from PIL import Image
import sift

ima_dir = "./IMG"
data_path = ima_dir + "/*.JPG"
files = glob.glob(data_path)
data = []
comp = cv2.imread("DSC01873.JPG", 0)
comp = cv2.resize(comp, dsize=(256, 256), interpolation=cv2.INTER_AREA)
print(comp)
kp, des, patches = sift.computeKeypointsAndDescriptors(comp)
cnt_file = []
cnt = []
for i in range(len(kp)):
    cnt_file.append(0)
    cnt.append(0)
sift_cv2 = cv2.xfeatures2d.SIFT_create()
for k in range(len(patches)):
    patches[k] = np.array(patches[k])
    # arr = patches[k].reshape(50,50)
    # im = Image.fromarray(arr.astype('uint8'),'L')
    print(type(patches[k]))
    im = Image.fromarray(patches[k],'L')
    im.save('myim.png')
    im = cv2.imread('myim.png')
    kp, des = sift_cv2.detectAndCompute(im, None)
    for fl in files:
        img = cv2.imread(fl, 0)
        img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_AREA)
        kp1, des1 = sift_cv2.detectAndCompute(img,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des, des1, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.3 * n.distance:
                matchesMask[i] = [1, 0]
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)
        img3 = cv2.drawMatchesKnn(im, kp, img, kp1, matches, None, **draw_params)
        plt.imshow(img3, ), plt.show()
        cnt[k] = len(matches)
    if cnt[k] > 0:
        cnt_file[k] = cnt[k]
    print(cnt_file[k])
    print("  cnt file   ")










