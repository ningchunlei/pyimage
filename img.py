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
    def remove(self,polyGraph,dr):
        for x,y in self.points:
            del polyGraph[x][y][dr]

    def getVertex(self):
        x_lx = -1;
        x_ly = -1;
        x_ry = -1;

        x_b_lx =-1
        x_b_ly =-1
        x_b_ry = -1

        y_ly = -1;
        y_lx = -1
        y_rx = -1

        y_b_ly = -1
        y_b_lx = -1
        y_b_rx = -1

        for x,y in self.points:
            if x_lx == -1:
                x_lx = x
            if x_b_lx == -1:
                x_b_lx = x
            if y_ly == -1:
                y_ly = y
            if y_bly == -1:
                y_b_ly = y

            if x_lx > x :
                x_lx = x
                x_ly = y
                x_ry = y
                continue
            if x_b_lx < x :
                x_b_ly = y
                x_b_lx = x
                x_b_ry = y
                continue
            if y_ly > y :
                y_ly = y
                y_lx = x
                y_rx = x
                continue
            if y_b_ly < y:
                y_b_ly = y
                y_b_lx = x
                y_b_rx = x
                continue

            if x_lx == x:
                if x_ly > y :
                    x_ly = y
                if x_ry < y:
                    x_ry = y

            if x_b_lx == x:
                if x_b_ly > y:
                    x_b_ly = y
                if x_b_ry < y :
                    x_b_ry = y

            if y_ly == y:
                if y_lx > x :
                    y_lx = x
                if y_rx < x :
                    y_rx = x
            if y_b_ly == y:
                if y_b_lx > x :
                    y_b_lx = x
                if y_b_rx < x :
                    y_b_rx = x
        ((x_lx,x_ly),(x_lx,x_ry),(x_b_lx,x_b_ly),(x_b_lx,x_b_ry)),((y_lx,y_ly),(y_b_lx,y_b_ly),(y_rx,y_ly),(y_b_rx,y_b_ly))

    def removeOthers(self,polyGraph,leftTop,rightTop,leftBottom,rightBottom):
        for x,y in self.points:
            if x >= leftTop[0] and  y>=leftTop[1] and x>=rightTop[0] and y<=rightTop[1] and x<=leftBottom[0] and y>=leftBottom[1] and x<=rightBottom[0] and  y<=rightBottom[1]:
                continue
            else:
                self.points.remove((x,y))
                del polyGraph[x][y][1]

    def build(self,polyGraph):
        (leftTop,rightTop,leftBottom,rightBottom),(_leftTop,_rightTop,_leftBottom,_rightBottom) = self.getVertex()
        if abs(dr)==1:
            if leftBottom[0]-leftTop[0] >= _leftBottom[0]-_leftTop[0] :
                self.removeOthers(polyGraph,leftTop,rightTop,leftBottom,rightBottom)
                self.length = leftBottom[0]-leftTop[0]
            else:
                self.removeOthers(polyGraph,_leftTop,_rightTop,_leftBottom,_rightBottom)
                self.length = _leftBottom[0]-_leftTop[0]
        elif abs(dr)==3:
            if rightTop[1]-leftTop[1] >= _rightTop[1] - _leftTop[1] :
                self.removeOthers(polyGraph,leftTop,rightTop,leftBottom,rightBottom)
                self.length = rightTop[1]-leftTop[1]
            else:
                self.removeOthers(polyGraph,_leftTop,_rightTop,_leftBottom,_rightBottom)
                self.length = _rightTop[1]-_leftTop[1]
        elif abs(dr)==2:
            self.length = leftBottom[0] - leftTop[0]
        elif abs(dr)==6:
            self.length = leftBottom[0] - leftTop[0]
        elif abs(dr)==4:
            self.length = leftBottom[0] - leftTop[0]
        elif abs(dr)==8:
            self.length = leftBottom[0] - leftTop[0]



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



