import errno
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from PIL import Image
import sift
import time
start = time.time()
def dataPatch():
    data_set = []
    # class DATA:
    #     patch = []
    #     label = 0
    #     def __init__(self, p, ll):
    #         patch = p
    #         label = ll
    for i in range(300):
        data_set.append([])
    data_set[0].append('Patch_num')
    data_set[0].append('Label')
    # for i in range(6):
    #     try:
    #         os.makedirs(str(i))
    #     except OSError as e:
    #         if e.errno != errno.EEXIST:
    #             raise
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
    patches_img = []

    sift_cv2 = cv2.xfeatures2d.SIFT_create()
    labels = []
    patches_name = 'patch_'
    file_extension = '.png'
    number = 0
    for k in range(len(patches)):
        patches[k] = np.array(patches[k])
        # arr = patches[k].reshape(50,50)
        # im = Image.fromarray(arr.astype('uint8'),'L')
        print(type(patches[k]))
        if patches[k].shape[0] != 0 and patches[k].shape[1] != 0:
            im = Image.fromarray(patches[k], 'L')
            im.save('myim.png')
            im = cv2.imread('myim.png')
            kp, des = sift_cv2.detectAndCompute(im, None)
            # im = im.astype('float32')
            patches_img.append(cv2.resize(im, dsize=(16, 16), interpolation=cv2.INTER_AREA))
            # p = []
            # p.append(np.array(patches).resize((k,16,16)))
            # print(p.__sizeof__())
            for fl in files:
                img = cv2.imread(fl, 0)
                img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
                kp1, des1 = sift_cv2.detectAndCompute(img, None)
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
            labels.append(cnt_file[k])
            dir_name = './'+str(cnt_file[k])+'/'
            im = Image.fromarray(patches[k], 'L')
            #im.save(dir_name + patches_name + str(number) + file_extension)

            data_set[number+1].append(dir_name + patches_name + str(number) + file_extension)
            data_set[number+1].append(cnt_file[k])
            number += 1
            print(cnt_file[k])
            print("time :", time.time() - start)
    return data_set
