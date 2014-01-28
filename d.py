# -*- coding: UTF-8 -*-
__author__ = 'Administrator'


import cv2
import numpy


img = numpy.zeros((700,700),numpy.uint8)
img.fill(255)

for x in xrange(255):
    for y in xrange(255):
        img[x+10][y+10]=y
        img[x+10][y+10+1]=y



cv2.imwrite("d:/g.jpg",img)

