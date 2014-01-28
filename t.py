__author__ = 'Administrator'
import numpy as np
import cv2

img = cv2.imread('d:/a.jpg',cv2.IMREAD_GRAYSCALE)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv2.imshow('image',cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()



