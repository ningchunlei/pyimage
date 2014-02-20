import cv2
import numpy
import scipy
from scipy import ndimage


class ImgPath(object):
    def __init__(self,x=0,y=0,direction=0):
        self.startX = x
        self.startY = y
        self.endX =x
        self.endY = y
        self.direction = abs(direction)
        self.points={}
        self.points[(x,y)]=(x,y)
        self.length = len(self.points)
        self.width = 0

    def setXY(self,x,y):
        if self.direction == 3 :
            if self.startY > y :
                self.startX = x
                self.startY = y
            elif self.endY < y :
                self.endX = x
                self.endY = y
        else:
            if self.startX > x :
                self.startX = x
                self.startY = y
            elif self.endX < x :
                self.endX = x
                self.endY = y
        self.points[(x,y)]=(x,y)
        self.length = len(self.points)

    def __str__(self):
        return "startX=%d,startY=%d,endX=%d,endY=%d,length=%d,direction=%d" % (self.startX,self.startY,self.endX,self.endY,self.length,self.direction)

class ImgSet(object):

    def __init__(self,index,dr):
        self.index = index
        self.direction = dr
        self.points = []

    def addPoint(self,x,y):
        self.points.append((x,y))

    def replaceImgSet(self,replace,polyGraph,dr):
        for x,y in replace.points:
            polyGraph[x][y].put(dr,self)
            self.addPoint(x,y)

class ImgPoint(object):

    def __init__(self,x=0,y=0,direction=0,nearzero=False):
        self.x = x
        self.y = y
        self.path=None
        self.direction = direction
        self.nearZero = nearzero

    def setPoint(self,x,y,direction,nearzero):
        self.x = x
        self.y = y
        self.direction = direction
        self.nearZero = nearzero



