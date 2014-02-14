import cv2
import numpy
import scipy
from scipy import ndimage
from img import ImgPath,ImgPoint
import copy

imgOrgin = cv2.imread('d:/a.jpg',cv2.IMREAD_GRAYSCALE)
row,column = imgOrgin.shape

copyOrgin = numpy.zeros(imgOrgin.shape,"int32")

counter=0;
for x in imgOrgin:
    copyOrgin[counter]=x
    counter += 1

pixel_mean = numpy.mean(copyOrgin)
pixel_histo = numpy.histogram(imgOrgin,bins=range(256))[0]

pixel_max = numpy.argmax(pixel_histo)

rightLevel = numpy.sum(pixel_histo[pixel_max:-1])
leftLevel = numpy.sum(pixel_histo[0:pixel_max])


grad_x = ndimage.filters.sobel(copyOrgin,0)
grad_y = ndimage.filters.sobel(copyOrgin,1)

grad_mag = numpy.sqrt(grad_x**2 + grad_y**2)
grad_angle = numpy.arctan2(grad_y, grad_x)


grad_mag_mean = numpy.mean(grad_mag)

grad_mag = grad_mag.astype(int)


grad_mask = grad_mag > grad_mag_mean
pixel_mask = None
if leftLevel > rightLevel:
    pixel_mask = imgOrgin > pixel_mean
else :
    pixel_mask = imgOrgin < pixel_mean


mask = ~(pixel_mask & grad_mask)
img = numpy.ma.array(imgOrgin,mask=mask,fill_value=0).filled()
img.dtype=numpy.uint8

mask = ~pixel_mask
extendImg = numpy.ma.array(imgOrgin,mask=mask,fill_value=0).filled()
extendImg.dtype=numpy.uint8



fillImg = numpy.insert(img,0,0,axis=0)
fillImg = numpy.insert(fillImg,0,0,axis=1)
fillImg = numpy.insert(fillImg,fillImg.shape[1],0,axis=1)
fillImg = numpy.insert(fillImg,fillImg.shape[0],0,axis=0)


extendImg = numpy.insert(extendImg,0,0,axis=0)
extendImg = numpy.insert(extendImg,0,0,axis=1)
extendImg = numpy.insert(extendImg,extendImg.shape[1],0,axis=1)
extendImg = numpy.insert(extendImg,extendImg.shape[0],0,axis=0)

r0 = -numpy.pi/8
r1 = -numpy.pi*3/8
r2 = -numpy.pi*5/8
r3 = -numpy.pi*7/8

r4 = numpy.pi*7/8
r5 = numpy.pi*5/8
r6 = numpy.pi*3/8
r7 = numpy.pi/8

r8 = -numpy.pi/2
r9 = numpy.pi/2


def findRedirection(x,y):
    x -=1
    y -=1
    if grad_y[x][y] < 0 :
        t = grad_angle[x][y]
        if t>=r0 and t<=0 :
            return -3
        elif t>=r1 and t<=r0:
            return -2
        elif t>=r8 and t<=r1:
            return 1
        elif t>=r2 and t<=r8:
            return -1
        elif  t>=r3 and t<=r2:
            return 4
        elif t>=-numpy.pi and t<=r3:
            return -3
    elif grad_y[x][y] >= 0:
        t = grad_angle[x][y]
        if t>=0 and t<=r7 :
            return 3
        elif t>=r7 and t<=r6:
            return 4
        elif t>=r6 and t<=r9:
            return 1
        elif t>=r9 and t<=r5:
            return -1
        elif  t>=r5 and t<=r4:
            return 2
        elif t>=r4 and t<=numpy.pi:
            return 3


dmap = {
    1:(1,0),
    2:(1,1),
    3:(0,1),
    4:(-1,1),
    -1:(-1,0),
    -2:(-1,-1),
    -3:(0,-1),
    -4:(1,-1)
}

td = numpy.ndarray(fillImg.shape,"int8")
td.fill(0)

startX=1
startY=1
endX = row+1
endY = column+1

for x in range(startX,endX,1):
    for y in range(startY,endY,1):
        if fillImg[x][y] == 0 :
            continue
        dr = findRedirection(x,y)
        td[x][y] = dr
        dx,dy = dmap[dr]
        if fillImg[x+dx][y+dy] !=0 :
            continue
        tmp = fillImg[x][y]
        fillImg[x][y]=0
        tdtmp = td[x][y]
        td[x][y]=0
        for index in dmap:
            dx,dy = dmap[index]
            if fillImg[x+dx,y+dy]==0:
                continue
            dr = findRedirection(x+dx,y+dy)
            if index + dr == 0:
                fillImg[x][y]=tmp
                td[x][y]=tdtmp
                break

graph = numpy.ndarray(fillImg.shape,"object")
polyGraph = numpy.ndarray(fillImg.shape,"object")

def fillGraph(x,y,dx,dy,flag_op,dx_op=0,dy_op=0):
    point = graph[x][y]
    if point == None :
        path = ImgPath(x,y,index)
        point = ImgPoint(x,y,index,td[x+dx][y+dy]==0)
        point.path = path
        graph[x][y] = point
    else:
        ImgPoint.setPoint(point,x,y,index,td[x+dx][y+dy]==0)
        ImgPath.setXY(point.path,x,y)
    nextpoint = None
    if flag_op == True :
        dx = dx_op
        dy = dy_op

    tmpPolyGraph = polyGraph[x,y]
    if tmpPolyGraph == None :
        tmpPolyGraph = set()
        polyGraph[x,y]=tmpPolyGraph
    tmpPolyGraph.add(point.path.direction)

    nextpoint = graph[x+dx][y+dy]
    if nextpoint == None:
        nextpoint = ImgPoint(x+dx,y+dy,td[x+dx][y+dy])
        nextpoint.path = point.path
        graph[x+dx][y+dy]=nextpoint


startX=1
startY=1
endX = 77
endY = column+1
for x in range(startX,endX):
    for y in range(startY,endY):
        index = td[x][y]
        if index == 0 :
            continue
        dx,dy = dmap[index]
        dx_op,dy_op = dmap[-index]
        if x ==15 :
            print 1
        if td[x+dx][y+dy] == index or td[x+dx][y+dy] == -index:
            fillGraph(x,y,dx,dy,False)
        if td[x+dx_op][y+dy_op] == index or td[x+dx_op][y+dy_op] == -index:
            fillGraph(x,y,dx,dy,True,dx_op,dy_op)


print "fillGraph"

"""
t1={}
for x in range(startX,endX):
    for y in range(startY,endY):
        if graph[x,y] != None:
            pt = graph[x,y].path
            t1[pt] = pt
rt = sorted(t1.keys(),key=lambda x:x.startX)
for v in rt:
    print str(v)

exit()
"""

def horizontalPath(path):
    pathMean = -1
    rowMean = -1
    dx,dy = dmap[td[path.startX][path.startY]]
    path=copy.copy(path)
    if abs(td[path.startX+dx][path.startY+dy]) != 3 and abs(td[path.startX+dx][path.startY+dy])>0:
        path.startX = path.startX+dx
        path.startY = path.startY + dy
        path.length += 1
    dx,dy = dmap[td[path.endX][path.endY]]
    if abs(td[path.endX+dx][path.endY+dy]) != 3 and abs(td[path.startX+dx][path.startY+dy])>0:
        path.endX = path.endX+dx
        path.endY = path.endY + dy
        path.length += 1

    def __hpath(counter,step):
        pathMean = -1
        while True:
            weight = 0
            rowMean = -1
            tmpIndex = {}
            for j in xrange(path.length):
                t_x = path.startX+counter
                t_y = path.startY+j
                tmpIndex[(t_x,t_y)] = (t_x,t_y)
                if extendImg[t_x][t_y] == 0 :
                    weight = path.length<<2
                    break
                pt = graph[t_x][t_y]
                if pt != None and pt.path.direction!=path.direction:
                    if pt.path.length>4:
                        weight = path.length << 2
                    else :
                        weight +=1
                if td[t_x][t_y] ==0 :
                    if rowMean == -1 : rowMean = numpy.mean(extendImg[t_x][path.startY:path.startY+path.length])
                    if pathMean == -1:  pathMean = numpy.mean(extendImg[path.startX][path.startY:path.startY+path.length])
                    if abs(rowMean-pathMean) > pathMean*0.05:
                        weight +=1
            if weight> path.length/3 :
                return
            if abs(counter)>path.length:
                return
            counter = counter + step
            for tx,ty in tmpIndex:
                tmpPolyGraph = polyGraph[tx,ty]
                if tmpPolyGraph == None :
                    tmpPolyGraph = set()
                    polyGraph[tx,ty]=tmpPolyGraph
                tmpPolyGraph.add(3)

    __hpath(-1,-1)
    __hpath(1,1)
    return

def verticalPath(path):
    pathMean = -1
    rowMean = -1

    dx,dy = dmap[td[path.startX][path.startY]]
    path=copy.copy(path)
    if abs(td[path.startX+dx][path.startY+dy]) != 1 and abs(td[path.startX+dx][path.startY+dy])>0:
        path.startX = path.startX+dx
        path.startY = path.startY + dy
        path.length += 1
    dx,dy = dmap[td[path.endX][path.endY]]
    if abs(td[path.endX+dx][path.endY+dy]) != 1 and abs(td[path.startX+dx][path.startY+dy])>0:
        path.endX = path.endX+dx
        path.endY = path.endY + dy
        path.length += 1

    def vp(counter,step):
        pathMean = -1
        while True:
            weight = 0
            rowMean = -1
            tmpIndex = {}
            for j in xrange(path.length):
                t_y = path.startY+counter
                t_x = path.startX+j
                tmpIndex[(t_x,t_y)] = (t_x,t_y)
                if extendImg[t_x][t_y] == 0 :
                    weight = path.length<<2
                    break
                pt = graph[t_x][t_y]
                if pt != None and pt.path.direction!=path.direction:
                    if pt.path.length>4:
                        weight = path.length << 2
                    else :
                        weight +=1
                if td[t_x][t_y] ==0 :
                    if rowMean == -1 :
                        tmpAll = 0
                        for k in xrange(path.length):
                            tmpAll += extendImg[path.startX+k][t_y]
                        rowMean = tmpAll/path.length
                    if pathMean == -1:
                        tmpAll = 0
                        for k in xrange(path.length):
                            tmpAll += extendImg[path.startX+k][path.startY]
                        pathMean = tmpAll/path.length
                    if abs(rowMean-pathMean) > pathMean*0.05:
                        weight +=1
            if weight> path.length/3 :
                return
            if abs(counter)>path.length:
                return
            counter = counter + step
            for tx,ty in tmpIndex:
                tmpPolyGraph = polyGraph[tx,ty]
                if tmpPolyGraph == None :
                    tmpPolyGraph = set()
                    polyGraph[tx,ty]=tmpPolyGraph
                tmpPolyGraph.add(1)

    vp(-1,-1)
    vp(1,1)
    return


def downRightPath(path,dr):
    pathMean = -1
    rowMean = -1


    path=copy.copy(path)
    dx,dy = dmap[td[path.endX][path.endY]]
    if abs(td[path.endX+dx][path.endY+dy]) != 2 and abs(td[path.startX+dx][path.startY+dy])>0:
        path.endX = path.endX+dx
        path.endY = path.endY + dy
        path.length += 1

    def __axis(path,counter,j):
        t_y=0;t_x=0
        if dr == 4 :
            t_x = path.startX-counter+j
            t_y = path.startY+counter+j
        if dr == 3 :
            t_x = path.startX+j
            t_y = path.startY+counter+j
        if dr == 1 :
            t_x = path.startX+counter+j
            t_y = path.startY+j
        if t_x < 0 :t_x = 0
        if t_x > extendImg.shape[0]-1 : t_x = 0
        if t_y <0 : t_y = 0
        if t_y > extendImg.shape[1]-1 : t_y = 0
        return t_x,t_y

    def __hpath(counter,step):
        pathMean = -1
        while True:
            weight = 0
            rowMean = -1
            tmpIndex = {}
            for j in xrange(path.length):
                t_x ,t_y = __axis(path,counter,j)
                tmpIndex[(t_x,t_y)]  = (t_x,t_y)
                if extendImg[t_x][t_y] == 0 :
                    weight = path.length<<2
                    break
                pt = graph[t_x][t_y]
                if pt != None and pt.path.direction!=path.direction:
                    if pt.path.length>4:
                        weight = path.length << 2
                    else :
                        weight +=1
                if td[t_x][t_y] ==0 :
                    if rowMean == -1 :
                        tmpAll = 0
                        for k in xrange(path.length):
                            _x,_y = __axis(path,counter,k)
                            tmpAll += extendImg[_x][_y]
                        rowMean = tmpAll/path.length
                    if pathMean == -1:
                        tmpAll = 0
                        for k in xrange(path.length):
                            print path.startX,path.startX+k,path.startY,path.startY+k,path.length
                            tmpAll += extendImg[path.startX+k][path.startY+k]
                        pathMean = tmpAll/path.length
                    if abs(rowMean-pathMean) > pathMean*0.05:
                        weight +=1
            if weight> path.length/3 :
                return
            if abs(counter)>path.length:
                return

            counter = counter + step
            for tx,ty in tmpIndex:
                tmpPolyGraph = polyGraph[tx,ty]
                if tmpPolyGraph == None :
                    tmpPolyGraph = set()
                    polyGraph[tx,ty]=tmpPolyGraph
                tmpPolyGraph.add(2)
    __hpath(-1,-1)
    __hpath(1,1)
    return

def upRightPath(path,dr):
    pathMean = -1
    rowMean = -1

    dx,dy = dmap[td[path.startX][path.startY]]
    path=copy.copy(path)
    if abs(td[path.startX+dx][path.startY+dy]) != 4 and abs(td[path.startX+dx][path.startY+dy])>0:
        path.startX = path.startX+dx
        path.startY = path.startY + dy
        path.length += 1

    def __axis(path,counter,j):
        t_y=0;t_x=0
        if dr == 2 :
            t_x = path.startX+counter+j
            t_y = path.startY+counter-j
        if dr == 3 :
            t_x = path.startX+j
            t_y = path.startY+counter-j
        if dr == 1 :
            t_x = path.startX+counter+j
            t_y = path.startY-j
        if t_x < 0 :t_x = 0
        if t_x > extendImg.shape[0] : t_x = 0
        if t_y <0 : t_y = 0
        if t_y > extendImg.shape[1] : t_y = 0
        return t_x,t_y

    def __hpath(counter,step):
        pathMean = -1
        while True:
            weight = 0
            rowMean = -1
            tmpIndex = {}
            for j in xrange(path.length):
                t_x ,t_y = __axis(path,counter,j)
                tmpIndex[(t_x,t_y)]  = (t_x,t_y)
                if extendImg[t_x][t_y] == 0 :
                    weight = path.length<<2
                    break
                pt = graph[t_x][t_y]
                if pt != None and pt.path.direction!=path.direction:
                    if pt.path.length>4:
                        weight = path.length << 2
                    else :
                        weight +=1
                if td[t_x][t_y] ==0 :
                    if rowMean == -1 :
                        tmpAll = 0
                        for k in xrange(path.length):
                            _x,_y = __axis(path,counter,k)
                            tmpAll += extendImg[_x][_y]
                        rowMean = tmpAll/path.length
                    if pathMean == -1:
                        tmpAll = 0
                        for k in xrange(path.length):
                            tmpAll += extendImg[path.startX+k][path.startY+k]
                        pathMean = tmpAll/path.length
                    if abs(rowMean-pathMean) > pathMean*0.05:
                        weight +=1
            if weight> path.length/3 :
                return
            if abs(counter)>path.length:
                return
            counter = counter + step
            for tx,ty in tmpIndex:
                tmpPolyGraph = polyGraph[tx,ty]
                if tmpPolyGraph == None :
                    tmpPolyGraph = set()
                    polyGraph[tx,ty]=tmpPolyGraph
                tmpPolyGraph.add(4)
    __hpath(-1,-1)
    __hpath(1,1)
    return



for x in range(startX,endX):
    for y in range(startY,endY):
        point = graph[x][y]
        if point == None:
            continue
        path = point.path
        if path == None:
            continue
        if x == 15 :
            print x
        if path.direction == 3 :
            horizontalPath(path)
        elif path.direction == 1:
            verticalPath(path)
        elif path.direction == 2:
            #downRightPath(path,1)
            #downRightPath(path,3)
            downRightPath(path,4)
        elif path.direction == 4 :
            #upRightPath(path,1)
            #upRightPath(path,3)
            upRightPath(path,2)



keepGraph = numpy.ndarray(fillImg.shape,"uint8")
keepGraph.fill(0)

def find5by5Gird(x,y):
    sx = x - 2;sy = y - 2;ex = x +2;ey=y+2
    if sx<0 : sx = 0
    if sy<0 : sy = 0
    if ex > extendImg.shape[0]-1 : ex = extendImg.shape[0]-1
    if ey > extendImg.shape[1]-1 : ey = extendImg.shape[1]-1
    for tx in range(sx,ex):
        for ty in range(sy,ey):
            if polyGraph[tx][ty]!=None :
                 return True
    return False

for x in range(startX,endX):
    for y in range(startY,endY):
        if x == 15 and extendImg[x][y]==118:
            print x,y
        if polyGraph[x][y]==None  :
            pass
            #if find5by5Gird(x,y) == True : keepGraph[x][y] = extendImg[x][y]
        else:
            keepGraph[x][y] = extendImg[x][y]




cv2.imshow('image',keepGraph)
cv2.waitKey(0)
cv2.destroyAllWindows()

fd = open("d:/img_12.txt","w")
for x in range(startX,endX):
    for y in range(startY,endY):
        fd.write("%3d\t" % (keepGraph[x][y]))
    fd.write("\n")
fd.close()


fd = open("d:/img_td.txt","w")
for x in range(startX,endX):
    for y in range(startY,endY):
        fd.write("%3d\t" % (td[x][y]))
    fd.write("\n")
fd.close()