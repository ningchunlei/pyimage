import cv2
from types import NoneType
import numpy
import scipy
from scipy import ndimage

imgOrgin = cv2.imread('d:/a.jpg',cv2.IMREAD_GRAYSCALE)

#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
#imgOrgin = clahe.apply(imgOrgin)

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


mask = ~(pixel_mask)

img = numpy.ma.array(imgOrgin,mask=mask,fill_value=0).filled()

img.dtype=numpy.uint8

startX=15
startY=297
endX = 45
endY = 322

fillImg = numpy.insert(img,0,0,axis=0)
fillImg = numpy.insert(fillImg,0,0,axis=1)
fillImg = numpy.insert(fillImg,fillImg.shape[1],0,axis=1)
fillImg = numpy.insert(fillImg,fillImg.shape[0],0,axis=0)




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
        #td[x][y]=0
        for index in dmap:
            dx,dy = dmap[index]
            if fillImg[x+dx,y+dy]==0:
                continue
            dr = findRedirection(x+dx,y+dy)
            if index + dr == 0:
                fillImg[x][y]=tmp
                td[x][y]=tdtmp
                break





cv2.imshow('image',fillImg)
cv2.waitKey(0)
cv2.destroyAllWindows()




fd = open("d:/img_12.txt","w")
for x in imgOrgin[startX-1:endX-1]:
    out = numpy.array_str(x[startY-1:endY-1],max_line_width=2000)
    fd.write(out+"\n")
fd.close()

fd = open("d:/img_1x.txt","w")
for x in grad_x[startX-1:endX-1]:
    out = numpy.array_str(x[startY-1:endY-1],max_line_width=2000)
    fd.write(out+"\n")
fd.close()

fd = open("d:/img_1y.txt","w")
for x in grad_y[startX-1:endX-1]:
    out = numpy.array_str(x[startY-1:endY-1],max_line_width=2000)
    fd.write(out+"\n")
fd.close()

fd = open("d:/img_1mag.txt","w")
for x in grad_mag[startX-1:endX-1]:
    out = numpy.array_str(x[startY-1:endY-1],max_line_width=2000)
    fd.write(out+"\n")
fd.close()

fd = open("d:/img_1ang.txt","w")
for x in grad_angle[startX-1:endX-1]:
    out = numpy.array_str(x[startY-1:endY-1],max_line_width=2000)
    fd.write(out+"\n")
fd.close()

fd = open("d:/img_td.txt","w")
for x in td[startX-1:endX-1]:
    out = numpy.array_str(x[startY-1:endY-1],max_line_width=2000)
    fd.write(out+"\n")
fd.close()





"""
fd = open("d:/img_a.txt","w")
for x in td[startX:endX]:
    out = numpy.array_str(x[startY:endY],max_line_width=2000)
    fd.write(out+"\n")
fd.close()
"""


"""
fd = open("d:/img1.txt","w")
for x in imgOrgin:
    out = numpy.array_str(x[0:200],max_line_width=2000)
    fd.write(out+"\n")
fd.close()


fd = open("d:/img2.txt","w")
for x in grad_mag:
    out = numpy.array_str(x[0:200],max_line_width=2000)
    fd.write(out+"\n")
fd.close()

fd = open("d:/img3.txt","w")
for x in grad_angle:
    out = numpy.array_str(x[0:200],max_line_width=2000)
    fd.write(out+"\n")
fd.close()

fd = open("d:/imgx.txt","w")
for x in grad_x:
    out = numpy.array_str(x[0:200],max_line_width=2000)
    fd.write(out+"\n")
fd.close()


fd = open("d:/imgy.txt","w")
for x in grad_y:
    out = numpy.array_str(x[0:200],max_line_width=2000)
    fd.write(out+"\n")
fd.close()

"""



















