import cv2
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
img = cv2.imread('./IMG/DSC01726.JPG')
img = cv2.resize(img, dsize=(600, 400), interpolation=cv2.INTER_AREA)
cv2.imshow("ddd.jpg", img)
cv2.waitKey()
cv2.destroyWindow()


img = cv2.imread('./IMG/DSC01726.JPG')
img = cv2.resize(img, dsize=(600, 400), interpolation=cv2.INTER_AREA)
rows,cols,ch = img.shape

pts1 = np.float32([[186,171],[314, 9],[487,106]])
pts2 = np.float32([[273,192],[350,83],[426, 149]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

# dir_name = './aff_IMG/'
# file_extension = '.png'
# dst = np.delete(dst, 1, 3)
# dst = np.delete(dst, 1, 3)
# dst = dst.reshape(600,400)
# im = Image.fromarray(dst, 'L')
# im.save(dir_name + '1726' + file_extension)

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()