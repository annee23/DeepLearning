#ref https://github.com/rmislam/PythonSIFT

from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
####################
# Global variables #
####################

logger = logging.getLogger(__name__)
float_tolerance = 1e-7

#################
# Main function #
#################
import time
start = time.time()
def main():
    logger = logging.getLogger(__name__)

    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('box.png', 0)           # queryImage
    img2 = cv2.imread('box_in_scene.png', 0)  # trainImage

    # Compute SIFT keypoints and descriptors
    kp1, des1 = func(img1)
    kp2, des2 = func(img2)

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Draw detected template in scene image
        h, w = img1.shape
        pts = np.float32([[0, 0],
                          [0, h - 1],
                          [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        plt.imshow(newimg)
        plt.show()
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

sigma = sqrt(2)
octave = 3
level = 3
row = col = 256
def func(image):
    image = image.astype('float32')
    base_image = generateBaseImage(image)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    print("time :", time.time() - start)
    gaussian_kernels = generateGaussianKernels(num_intervals=3)
    print("time :", time.time() - start)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals=3, image_border_width=5)
    print("time :", time.time() - start)
    #keypoints = removeDuplicateKeypoints(keypoints)
    #keypoints = convertKeypointsToInputImageSize(keypoints)
    #descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints#,descriptors
#########################
# Image pyramid related #
#########################

def generateBaseImage(image, assumed_blur=0.5):
    """Generate base image from input image by upsampling by 2 in both directions and blurring
    """
    logger.debug('Generating base image...')
    image = resize(image, dsize=(256, 256), interpolation=INTER_LINEAR)
    # image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff,
                        sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur

def computeNumberOfOctaves(image_shape):
    """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
    """
    return int(round(log(min(image_shape)) / log(2) - 1))

def generateGaussianKernels(num_intervals):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
    """
    logger.debug('Generating scales...')
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = zeros(
        num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """Generate scale-space pyramid of Gaussian images
    """
    logger.debug('Generating Gaussian images...')
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
        for gaussian_kernel in gaussian_kernels[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                       interpolation=INTER_NEAREST)
    return array(gaussian_images)

def generateDoGImages(gaussian_images):
    """Generate Difference-of-Gaussians image pyramid
    """
    logger.debug('Generating Difference-of-Gaussian images...')
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(subtract(second_image,
                                                 first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return array(dog_images)

###############################
# Scale-space extrema related #
###############################
# Keypoint Localistaion ####search each pixel in the DoG map to find the extreme point


def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, image_border_width):
    interval = level - 1
    number = 0
    for i in range(octave - 1):
        number = number + (2 ** (i - octave) * col) * (2 * row) * interval
    extrema = []
    flag = 1
    for i in range(octave):
        m = len(dog_images[i])
        n = len(dog_images[i][0])
        m = m - 2
        n = n - 2
        volume = m * n // (4 ** i)
        for k in range(interval - 1):
            for j in range(volume):
                x = math.ceil((j + 1) / n)
                y = j % m + 1
                sub = dog_images[i][x - 1:x + 1, y - 1:y + 1, k - 2:k]
                lar = max(max(max(sub)))
                lit = min(min(min(sub)))
                if (lar == dog_images[i][x, y, k - 1]):
                    temp = [i - 1, k - 1, j - 1, 0]
                    extrema[flag - 1:flag + 2] = temp
                    flag = flag + 4
                if (lit == dog_images[i][x, y, k - 1]):
                    temp = [i - 1, k - 1, j - 1, -1]
                    extrema[flag - 1:flag + 2] = temp
                    flag = flag + 4

    idx = (extrema == 0)
    extrema[idx] = []
    m = n = 256
    x = math.floor((extrema[2:3:-1] - 1) // (n // (2 ** (extrema[0:3:-1] - 2)))) + 1
    y = (extrema[2:3:-1] - 1) % (m // (2 ** (extrema[0:3:-1] - 2))) + 1
    ry = y // 2 ** (octave - 1 - extrema[0:3:-1])
    rx = x // 2 ** (octave - 1 - extrema[0:3:-1])


# accurate keypoint localization ###### eliminate the point with low contrast or localised on edge

    threshold = 0.1
    r = 10
    extr_volume = len(extrema)//4
    m = n = 256
    secondorder_y = conv2([-1,-1][1,1],[-1,-1][1,1])
    for i in range(octave):
        for j in range(level):
            test = dog_images[i][::j]
            temp = -1//conv2(test,secondorder_y,'same')*conv2(test,[-1,-1][1,1],'same')
            dog_images[i][::j] = temp*conv2(test,[-1,-1][1,1],'same')*0.5+test
    for i in range(extr_volume):
        x = math.floor((extrema[4*(i)+3]-1)//(n//(2**(extrema[4*(i)+1]-2))))+1
        y = (extrema[4*(i)+3]-1)%(m//2**(extrema[4*(i)+1]-2))+1
        rx = x+1
        ry = y +1
        rz = extrema[4*(i)+2]
        z = dog_images[extrema[4*i+1]][rx,ry,rz]
        if (math.fabs(z)<threshold):
            extrema[4*i+4]=0
    idx = [k for (k,val) in enumerate(extrema) if val==0]
    idx = [idx,idx-1,idx-2,idx-3]
    extrema[idx]=[]
    extr_volume  = len(extrema)//4
    x = math.floor((extrema[2:3:-1]-1)//(n//(2**(extrema[0:3:-1]-2))))+1
    y = (extrema[2:3:-1]-1)%(m//(2**(extrema[0:3:-1])))+1
    ry = y//2**(octave-1-extrema[0:3:-1])
    rx = x//2**(octave-1-extrema[0:3:-1])

    for i in range(extr_volume):
        x = floor((extrema[4*i+3]-1)//(n//2**(extrema[4*i+1]-2)))+1
        y = (extrema[4*i+3]-1)%(m//2**(extrema[4*i+1]-2))+1
        rx = x
        ry = y
        rz = extrema(4*i+2)-1
        Dxx = dog_images[extrema[4 * i + 1]][rx - 1, ry, rz] + dog_images[extrema[4 * (i) + 1]][rx + 1, ry, rz] - 2 *  dog_images[extrema[4 * (i) + 1]][rx, ry, rz]
        Dyy = dog_images[extrema[4 * (i ) + 1]][rx, ry - 1, rz] + dog_images[extrema[4 * (i ) + 1]][rx, ry + 1, rz] - 2 * dog_images[extrema[4 * (i) + 1]][rx, ry, rz]
        Dxy = dog_images[extrema[4 * (i ) + 1]][rx - 1, ry - 1, rz] + dog_images[extrema[4 * i + 1]][rx + 1, ry + 1, rz] - dog_images[extrema[4 * (i ) + 1]][rx - 1, ry + 1, rz] - dog_images[extrema[4 * (i) + 1]][rx + 1, ry - 1, rz]
        deter = Dxx * Dyy - Dxy * Dxy
        R = (Dxx + Dyy) // deter
        R_threshold = (r + 1) **2 / r
        if (deter < 0 or R > R_threshold):
            extrema[4 * i + 4] = 0
    idx = [k for (k, val) in enumerate(extrema) if val == 0]
    idx = [idx,idx - 1, idx - 2, idx - 3]
    extrema[idx] = []
    extr_volume = len(extrema) // 4
    x = math.floor((extrema[2:3:-1] - 1) // (n // (2 ** (extrema[0:3:-1] - 2)))) + 1
    y = (extrema[2:3:-1] - 1) % (m // (2 ** (extrema[0:3:-1]))) + 1
    ry = y // 2 ** (octave - 1 - extrema[0:3:-1])
    rx = x // 2 ** (octave - 1 - extrema[0:3:-1])
    ##########################################################
    #Orienation Assignmaent
    kpori = []
    minor = []
    f =1
    flag =1
    #for i in range(extr_volume):
       # scale = sigma * math.sqrt(2)**(1//level)**((extrema[4*i+1]-1)*level+(extrema[4*i+2]))
      #  width = 2*round(3*1.5*scale)
      #  count = 1
       # x = math.floor((extrema[4*i+3]-1)//(n//2**(extrema[4*i+1]-2)))+1
       # y = (extrema[4*i+3]-1)%m//2**(extrema[4*i+1]-2)+1
    return extrema










########################################################################################################################




if __name__ == "__main__":
    main()