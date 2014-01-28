import cv2
import numpy
import scipy

img1 = cv2.imread('d:/a.jpg')
imgOrgin = cv2.imread('d:/a.jpg',cv2.IMREAD_GRAYSCALE)

img = cv2.Canny(imgOrgin,100,200)

row = img.shape[0]
column = img.shape[1]


"""
row = img.shape[0]
column = img.shape[1]

xaxis=[0] * row

for x in xrange(row):
    tmp = 0;
    for y in  xrange(column):
        b =0;
        if img[x,y]==255 :
            b = 1
        tmp += b
    xaxis[x] = tmp

xheigh = []
start = -1
for i,v in enumerate(xaxis):
      if start == -1  and v>5 :
          start = i;
      elif  start>=0 and v<=5:
          xheigh.append(i-start)
          start = -1

print xheigh
"""

def tracePath(row,start,direction,img,step,maxStep):
    right = lambda start,end : start < end
    left = lambda start,end : start > end
    ld=left
    end = 0
    if step>0:
        end = img[row].shape[0]
        ld = right
    start = start + step
    counter = 0
    while  ld(start,end):
        if direction>0 and img[row][start]>=0:
            start = start + step
        elif direction<0 and img[row][start]<=0:
            start = start + step
        else:
            break;
        counter += 1
        if counter > maxStep :
            break
    return start

def findFirstNotZero(row,start,step,img):
    end = img[row].shape[0]
    while start < end :
        if img[row][start]==0:
            start += 1
        else :
            break
    return start

def findNext(start,row,column,left,right):
    max = row.shape[0]
    if start >= max:
        return start

    startRow = row[start]
    nextStart = start + 1;
    while  nextStart < max:
        nextRow = row[nextStart]
        nextColumn = column[nextStart]
        if startRow == nextRow and left <= nextColumn and nextColumn <= right:
            nextStart += 1
        else:
            break
    return nextStart


"""
imgdiff = numpy.diff(imgOrgin)
imgdiff.dtype=numpy.int8
drow,dcolumn =  img.nonzero()
length = drow.shape[0]
start=0;
maxStep = 0
guessImg = numpy.zeros(img.shape,numpy.uint8)

while start < length:
    startColumn = dcolumn[start]
    startRow = drow[start]
    guessImg[startRow][startColumn] = img[startRow][startColumn]
    start += 1

    startDirection = imgdiff[startRow][startColumn]
    if startDirection == 0:
        tmp = findFirstNotZero(startRow,startColumn,1,imgdiff)
        if tmp +1 == imgdiff[startRow].shape[0] :
            start = start +1
            continue
        else:
            startDirection = imgdiff[startRow][tmp]
    leftEdge = tracePath(startRow,startColumn,startDirection,imgdiff,-1,maxStep)
    rightEdge = tracePath(startRow,startColumn,startDirection,imgdiff,1,maxStep)
    for t in xrange(rightEdge-leftEdge+1):
        guessImg[startRow][leftEdge+t]=255

    start = findNext(start,drow,dcolumn,leftEdge,rightEdge)

"""




















#for x in range(43,100):
#    for y in xrange(80):
#imgrs=25
#imgre=100

#imges=40
#imgee=90


imgrs=82
imgre=135

imges=834
imgee=902


t = img[imgrs:imgre]
filpImg = numpy.zeros((imgre-imgrs,imgee-imges),numpy.uint8)

counter=0;
for x in t:
    filpImg[counter]=x[imges:imgee]
    counter += 1

t = imgOrgin[imgrs:imgre]
filpOrgin = numpy.zeros((imgre-imgrs,imgee-imges),"int32")

counter=0;
for x in t:
    filpOrgin[counter]=x[imges:imgee]
    counter += 1



fd = open("d:/a.txt","w")
for x in filpOrgin:
    out = numpy.array_str(x,max_line_width=2000)
    fd.write(out+"\n")
fd.close()

from scipy import ndimage

md = ndimage.filters.median_filter(filpOrgin,3)

fd = open("d:/ab.txt","w")
for x in md:
    out = numpy.array_str(x,max_line_width=2000)
    fd.write(out+"\n")
fd.close()

grad_x = ndimage.filters.sobel(filpOrgin,0)

fd = open("d:/sx.txt","w")
for x in grad_x:
    out = numpy.array_str(x,max_line_width=2000)
    fd.write(out+"\n")
fd.close()

grad_y = ndimage.filters.sobel(filpOrgin,1)

fd = open("d:/sy.txt","w")
for x in grad_y:
    out = numpy.array_str(x,max_line_width=2000)
    fd.write(out+"\n")
fd.close()

grad_mag = numpy.sqrt(grad_x**2 + grad_y**2)
grad_angle = numpy.arctan2(grad_y, grad_x)
#grad_angle = grad_angle.astype(int)
grad_mag = grad_mag.astype(int)

fd = open("d:/mag.txt","w")
for x in grad_mag:
    out = numpy.array_str(x,max_line_width=2000)
    fd.write(out+"\n")
fd.close()


fd = open("d:/angle.txt","w")
for x in grad_angle:
    out = numpy.array_str(x,max_line_width=2000)
    fd.write(out+"\n")
fd.close()




#print filpOrgin

grad_mag = grad_mag.astype("uint8")
print numpy.mean(imgOrgin)

print numpy.histogram(imgOrgin,bins=range(256))

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite("d:/b1.jpg",grad_mag)



