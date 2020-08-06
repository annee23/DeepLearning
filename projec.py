import cv2
from PIL import Image
import numpy as np

#This will display all the available mouse click events
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

#This variable we use to store the pixel location
refPt = []
pnt = []
pnt2 = []
#click event function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pnt.append([x, y])
        print(x,",",y)
        refPt.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)
        cv2.putText(img, strXY, (x,y), font, 0.5, (255,255,0), 2)
        cv2.imshow("image", img)

    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue)+", "+str(green)+","+str(red)
        cv2.putText(img, strBGR, (x,y), font, 0.5, (0,255,255), 2)
        cv2.imshow("image", img)

def click_event2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pnt2.append([x, y])
        print(x,",",y)
        refPt.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)
        cv2.putText(img2, strXY, (x,y), font, 0.5, (255,255,0), 2)
        cv2.imshow("image", img2)

    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img2[y, x, 0]
        green = img2[y, x, 1]
        red = img2[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue)+", "+str(green)+","+str(red)
        cv2.putText(img2, strBGR, (x,y), font, 0.5, (0,255,255), 2)
        cv2.imshow("image", img2)




#Here, you need to change the image name and it's path according to your directory
img = cv2.imread("DSC01756.JPG")
img = cv2.resize(img, dsize=(600, 400), interpolation=cv2.INTER_AREA)
cv2.imshow("image", img)

#calling the mouse click event
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = cv2.imread("./IMG/DSC01755.JPG")
img2 = cv2.resize(img2, dsize=(600, 400), interpolation=cv2.INTER_AREA)
cv2.imshow("image", img2)

cv2.setMouseCallback("image", click_event2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('./IMG/DSC01755.JPG', 0)
img = cv2.resize(img, dsize=(600, 400), interpolation=cv2.INTER_AREA)
rows, cols = img.shape[:2]
print(pnt)
src_points = np.float32(pnt2)
dst_points = np.float32(pnt)
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))

dir_name = './aff_IMG/'
file_extension = '.png'
# img_output = cv2.COLOR_RGB2GRAY(np.float32(img_output))
im = Image.fromarray(img_output, 'L')
im.save(dir_name + '1755' + file_extension)

cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey()