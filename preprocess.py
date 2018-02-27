#coding=utf-8
import cv2
import numpy as np
import math
import os
import copy
from itertools import combinations
import Levenshtein

class LineParam:
    def __init__(self, linept):
        self.x1 = linept[0]
        self.y1 = linept[1]
        self.x2 = linept[2]
        self.y2 = linept[3]
        self.p1 = [self.x1, self.y1]
        self.p2 = [self.x2, self.y2]
        p1=[self.x1, self.y1]
        p2=[self.x2, self.y2]
        line = []
        if p1[1] == p2[1]:
            theta = math.pi / 2
        else:
            theta = math.atan((float)(p2[0] - p1[0]) / (float)(p1[1] - p2[1]))
            if theta < 0:
                theta += math.pi
        rho = p1[0] * np.cos(theta) + p1[1] * np.sin(theta)
        line.append(rho)
        line.append(theta)
        self.line = line
        self.length = math.sqrt((self.x1-self.x2)**2+(self.y1-self.y2)**2)
        self.distanceU = 99999
        self.distanceD = 99999

    def getDistPtToLine(self, x, y):
        # 距离为正表示点在竖线右边，在横线下边
        # 当直线r<0时，distance加负号
        rho = self.line[0]
        theta = self.line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        r = x * a + y * b
        distance = r - rho
        if rho<0 :
            distance=0-distance
        return distance

    def drawLine(self, img):
        rho = self.line[0]
        theta = self.line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img

    def drawLineP(self, img):
        cv2.circle(img,(self.x1,self.y1),5,(0,0,255),2)
        cv2.circle(img,(self.x2,self.y2),5,(0,0,255),2)
        cv2.line(img, (self.x1,self.y1), (self.x2, self.y2), (0, 0, 255), 2)
        return img

    def getX(self, y):
        cos = np.cos(self.line[1])
        if np.abs(cos-0)<0.001:
            return self.p1[0]
        else:
            x = (self.line[0]- y*np.sin(self.line[1]))/cos
            return (int)(x)

    def getY(self,x):
        sin = np.sin(self.line[1])
        if np.abs(sin-0)<0.001:
            return self.p1[1]
        else:
            y = (self.line[0]- x*np.cos(self.line[1]))/sin
            return (int)(y)

def getLine(theta,pt):
    alpha = np.pi - theta
    rho = pt[0]*np.cos(alpha)+pt[1]*np.sin(alpha)
    myline = LineParam([pt[0],pt[1],0,0])
    myline.line[0]=rho
    myline.line[1]=alpha
    return myline

def drawLine(lines,img):
    for line in lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return img

def drawLineP(lines,img):
    line = lines[0]
    for pts in line:
        # 自动补齐的线初始化时只有一个点，和theta rho参数
        if pts[2]!=0 and pts[3]!=0:
            cv2.line(img, (pts[0], pts[1]), (pts[2], pts[3]), (255, 0, 0), 2)
    return img

def lineCrossPoint(Line1,Line2):
    point = []
    line1=Line1.line
    line2=Line2.line
    if line1[1]==line2[1]:
        return []
    r = np.array([line1[0],line2[0]]).reshape(2,1)
    r = np.mat(r)
    c1=np.cos(line1[1]);s1=np.sin(line1[1])
    c2=np.cos(line2[1]);s2=np.sin(line2[1])
    a = np.array([c1,s1,c2,s2]).reshape(2,2)
    a=np.mat(a)
    xy = a.I * r
    point.append((int)(xy[0]))
    point.append((int)(xy[1]))
    return point

def LineVecHorDivide(lines, pts, angThresh):
    # 从直线组中筛选最接近水平和垂直的线条，并统计点和直线位置关系
    vecLines = []
    HorLines = []
    angThresh = (float)(angThresh)/180 * math.pi
    for line in lines:
        if line[1] < angThresh or math.fabs(line[1]-math.pi)<angThresh:# 垂直方向筛选
            myline = LineParam(line)
            myline.calculateParam(pts)
            vecLines.append(myline)
        elif math.fabs(line[1]-math.pi/2):# 水平方向筛选
            myline = LineParam(line)
            myline.calculateParam(pts)
            HorLines.append(myline)
    return vecLines, HorLines

def houghLine(img, threshold=150, minLength=150, maxGap=15):
    imgW = img.shape[1]
    imgH = img.shape[0]
    image = copy.copy(img)
    # brand
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=3)
    edges = cv2.Canny(edges,100,200)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, minLineLength=minLength, maxLineGap=maxGap)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # drawLineP(lines, edges)
    print "line num = ", len(lines[0])

    # 找垂线和横线
    VecLineL = None;
    VecLineR = None
    HorLineU = None;
    HorLineD = None
    angThresh = (float)(15)/(float)(180)*np.pi
    lenScaleThresh = 0.2
    veclines=[]
    horlines=[]
    for line in lines[0]:
        myline = LineParam(line)
        if myline.line[1] < angThresh or math.fabs(myline.line[1]-math.pi)<angThresh:# 垂直方向筛选
            if myline.length>=lenScaleThresh*imgH:
                veclines.append(myline)
        elif math.fabs(myline.line[1]-math.pi/2)<angThresh:# 水平方向筛选
            if myline.length>=lenScaleThresh*imgW:
                horlines.append(myline)

    # 画线
    for line in veclines:
        cv2.line(edges,tuple(line.p1),tuple(line.p2),(255,0,0),2)
    for line in horlines:
        cv2.line(edges,tuple(line.p1),tuple(line.p2),(255,2,2),2)

    # 直线按长度从大到小排序
    veclines.sort(key=lambda x:x.length,reverse=True)
    horlines.sort(key=lambda x:x.length,reverse=True)

    # 垂线更容易找，先确定垂线，根据垂线端点筛选横线
    if len(veclines)<0:
        print '没有找到垂线边缘，无法继续处理！'
        return img,None
    for line in veclines:
        if VecLineL==None and math.fabs(line.line[0])< (float)(imgW)/2:
            VecLineL=line
        if VecLineR==None and math.fabs(line.line[0])>(float)(imgW)/2:
            VecLineR=line


    # 找离垂线端点近的横线，横线应该左右垂线之间
    if len(horlines)<0:
        print '没有找到横线，无法继续处理！'
        return img, None
    p1s=[] # 垂线上端点
    p2s=[] # 垂线下端点
    VecLine=[]
    if VecLineL!=None:
        VecLine.append(VecLineL)
    if VecLineR!=None:
        VecLine.append(VecLineR)
    VecLine.sort(key=lambda x:x.length,reverse=True)
    if len(VecLine)<1:
        print '没有找到两侧的垂直边界线，中断！'
        return edges, image
    else:
        for vecline in VecLine:
            if vecline.y1<=vecline.y2:
                p1s.append(vecline.p1)
                p2s.append(vecline.p2)
            else:
                p2s.append(vecline.p1)
                p1s.append(vecline.p2)

    p1s.sort(key=lambda x:x[1])
    p2s.sort(key=lambda x:x[1], reverse=True)
    uplineThreshY = p2s[0][1]
    downlineThreshY = p1s[0][1]

    upLines = []
    downLines=[]
    maxLength = horlines[0].length
    for line in horlines:
        # if line.line[0]<(float)(imgH/2) and line.length>0.3*maxLength:
        if line.line[0]<(float)(uplineThreshY) and line.length>0.3*maxLength:
            # line.distance = math.fabs(line.getDistPtToLine(p1[0],p1[1]))
            distanceTmp = 0;numTmp=0
            for i in range((len(p1s))):
                distanceTmp+= math.fabs(line.getDistPtToLine(p1s[i][0],p1s[i][1]))
                numTmp+=1
            line.distanceU = distanceTmp/numTmp
            upLines.append(line)
        # if line.line[0]>(float)(imgH/2) and line.length>0.3*maxLength:
        if line.line[0]>(float)(downlineThreshY) and line.length>0.3*maxLength:
            # line.distance = math.fabs(line.getDistPtToLine(p2[0], p2[1]))
            distanceTmp = 0;
            numTmp = 0
            for i in range((len(p2s))):
                distanceTmp += math.fabs(line.getDistPtToLine(p2s[i][0], p2s[i][1]))
                numTmp += 1
            line.distanceD = distanceTmp / numTmp
            downLines.append(line)

    # 剔除不在左右垂线之间的线段
    UpLines = []
    DownLines = []
    deltpixel = 10
    if len(VecLine)==2:
        for line in upLines:
            if min(line.p1[0],line.p2[0])>=max(0, (min(p1s[0][0],p1s[1][0])-deltpixel)) \
                    and max(line.p1[0],line.p2[0])<=min(imgW,(max(p1s[0][0],p1s[1][0])+deltpixel)):
                UpLines.append(line)
        for line in downLines:
            if min(line.p1[0],line.p2[0])>=max(0,(min(p2s[0][0],p2s[1][0])-deltpixel)) \
                    and max(line.p1[0],line.p2[0])<=min(imgW,(max(p2s[0][0],p2s[1][0])+deltpixel)):
                DownLines.append(line)
    elif VecLineL==None:
        for line in upLines:
            if max(line.p1[0],line.p2[0])<=min(imgW,(p1s[0][0]+deltpixel)):
                UpLines.append(line)
        for line in downLines:
            if max(line.p1[0],line.p2[0])<=min(imgW,(p2s[0][0]+deltpixel)):
                DownLines.append(line)
    elif VecLineR==None:
        for line in upLines:
            if min(line.p1[0],line.p2[0])>=max(0, (p1s[0][0]-deltpixel)):
                UpLines.append(line)
        for line in downLines:
            if min(line.p1[0],line.p2[0])>=max(0,(p2s[0][0]-deltpixel)):
                DownLines.append(line)

    UpLines.sort(key=lambda x:x.distanceU)
    DownLines.sort(key=lambda x:x.distanceD)
    if len(upLines)>0:
        HorLineU=UpLines[0]
    if len(downLines)>0:
        HorLineD=DownLines[0]

    # 找直线交点
    Lines = []  # 顺序：左右上下
    if VecLineL!=None:
        Lines.append(VecLineL)
    if VecLineR!=None:
        Lines.append(VecLineR)
    if HorLineU!=None:
        Lines.append(HorLineU)
    if HorLineD!=None:
        Lines.append(HorLineD)

    # 如果有直线缺失，自动补齐 =======================================
    if len(Lines)==3:
        # 垂直边界是靠对称补齐
        if VecLineL==None:
            print '自动补齐左边界'
            pts = [HorLineD.p1, HorLineD.p2, HorLineU.p1, HorLineU.p2]
            pts.sort(key=lambda x: x[0])
            VecLineL = getLine(VecLineR.line[1], pts[0])
            Lines.append(VecLineL)
        if VecLineR==None:
            print '自动补齐右边界'
            pts = [HorLineD.p1, HorLineD.p2, HorLineU.p1, HorLineU.p2]
            pts.sort(key=lambda x:x[0],reverse=True)
            VecLineR = getLine(VecLineL.line[1],pts[0])
            Lines.append(VecLineR)

        # 水平边界补齐靠两条垂直边界端点
        deltScale = 0.067 # 40/600
        if HorLineU==None:
            print '自动补齐上边界'
            pts = [VecLineL.p1, VecLineL.p2, VecLineR.p1, VecLineR.p2]
            pts.sort(key=lambda x:x[1])
            if math.fabs(pts[0][1]-pts[1][1])<= deltScale*imgH:
                HorLineU = LineParam([pts[0][0],pts[0][1],pts[1][0],pts[1][1]])
            Lines.append(HorLineU)
        if HorLineD==None:
            print '自动补齐下边界'
            pts = [VecLineL.p1, VecLineL.p2, VecLineR.p1, VecLineR.p2]
            pts.sort(key=lambda x: x[1], reverse=True)
            if math.fabs(pts[0][1] - pts[1][1]) <= deltScale * imgH:
                HorLineD = LineParam([pts[0][0],pts[0][1],pts[1][0],pts[1][1]])
            Lines.append(HorLineD)

    # 完成直线补齐 =================================================

    # 画图保存直线提取结果
    for line in Lines:
        edges = line.drawLineP(edges)

    # 计算直线交点
    crossPoint = []
    for twolines in list(combinations(Lines, 2)):
        pt = lineCrossPoint(twolines[0], twolines[1])
        # 如果点超过image边界，扩充图像, 只向右边和下边扩充，不影响直线参数
        if len(pt)>0 and pt[0] > imgW and pt[0]< 1.5*imgW and pt[1]>=0 and pt[1]< 1.5*imgH:
            edgesTmp = np.zeros((imgH,pt[0]),np.uint8)
            edgesTmp[0:imgH,0:imgW] = edges
            edges = edgesTmp
            imageTmp = np.zeros((imgH,pt[0],3),np.uint8)
            imageTmp[0:imgH,0:imgW]=image
            image = imageTmp
            imgW = pt[0]
        if len(pt)>0 and pt[1] > imgH and pt[1]< 1.5*imgH and pt[0]>=0 and pt[0]< 1.5*imgW:
            edgesTmp = np.zeros((pt[1],imgW,3),np.uint8)
            edgesTmp[0:imgH,0:imgW] = edges
            edges = edgesTmp
            imageTmp = np.zeros((pt[1],imgW,3),np.uint8)
            imageTmp[0:imgH,0:imgW]=image
            image = imageTmp
            imgH = pt[1]
        if len(pt) > 0 and pt[0] <= imgW and pt[0] >= 0 and pt[1] >= 0 and pt[1] <= imgH:
            crossPoint.append(pt)
            cv2.circle(edges, (pt[0], pt[1]), 8, (0, 255, 255), 3)


    # 矫正图像
    cornerPoint=[]

    if len(crossPoint)<4:
        print '角点不足： ',len(crossPoint)
        return edges,image
    crossPoint.sort(key=lambda x:x[1])# y值小到大
    temp = [crossPoint[0], crossPoint[1]]
    temp.sort() # x值小到大
    cornerPoint.append(temp[0]) # 左上角点
    cornerPoint.append(temp[1]) # 右上角点
    temp = [crossPoint[2], crossPoint[3]]
    temp.sort() # x值小到大
    cornerPoint.append(temp[1]) # 右下角点
    cornerPoint.append(temp[0]) # 左下角点
    X1=(int)(cornerPoint[0][0]+cornerPoint[3][0])/2
    X2=(int)(cornerPoint[1][0]+cornerPoint[2][0])/2
    Y1=(int)(cornerPoint[0][1]+cornerPoint[1][1])/2
    Y2=(int)(cornerPoint[2][1]+cornerPoint[3][1])/2
    desPoint=[]
    desPoint.append([X1,Y1])
    desPoint.append([X2,Y1])
    desPoint.append([X2,Y2])
    desPoint.append([X1,Y2])

    # 透视变换 cornerPoint 纠正到 desPoint
    cornerPoint = np.float32(cornerPoint)
    desPoint = np.float32(desPoint)
    M = cv2.getPerspectiveTransform(cornerPoint,desPoint)
    newImg = cv2.warpPerspective(image,M,(imgW,imgH))
    # 裁剪
    delt = 25
    cropImg = newImg[max(0,Y1-delt):min(600,Y2+delt),max(0,X1-delt):min(800,X2+delt)]

    return edges,newImg



def main_line_restrain():
    imgs = 'image'
    goodPath = 'result'
    difficultPath = 'result'
    savepath = 'difficult/result'
    # single test
    # image = cv2.imread('difficultbkp/WP0AA298XJS260398.jpg')
    # image = cv2.resize(image,(800,600))
    # result,imageRestrain=houghLine(image)
    # cv2.imshow('result',result)
    # cv2.imshow('imageRefine', imageRestrain)
    # cv2.waitKey(0)
    #
    #
    count = 0
    for img in os.listdir(imgs):
        print 'image nums :',count
        count+=1

        print 'image=== ', img
        try:
            image = cv2.imread(os.path.join(imgs,img))
            image = cv2.resize(image,(800,600))

            boxResult,imageRestrain = houghLine(image,threshold=100,minLength=150,maxGap=20)
            cv2.imshow('result',boxResult)
            cv2.imshow('imageRefine',imageRestrain)
            key = cv2.waitKey(0)
            if key == 32:
                cv2.imwrite(os.path.join(difficultPath,img),image)
                cv2.imwrite(os.path.join('result',img),imageRestrain)
                cv2.imwrite(os.path.join( 'edges',img), boxResult)
                print '困难样本已储存：',img
                continue
            if key>0 and key!=32:
                cv2.imwrite(os.path.join(goodPath, img), image)
                cv2.imwrite(os.path.join( 'result',img),imageRestrain)
                cv2.imwrite(os.path.join( 'edges', img), boxResult)
                continue
        except:
            print 'erro!!!!'


# ===================================================================================== ctpn 后处理
def convert(img):
    rimg = np.zeros(tuple(img.shape),np.uint8)
    for i,row in enumerate(img):
        for j,col in enumerate(row):
            if col.size ==3:
                rimg[i][j][0]=255-col[0]
                rimg[i][j][1]=255-col[1]
                rimg[i][j][2]=255-col[2]
            elif col.size == 1:
                rimg[i][j] = 255 - col
    return rimg

def isWhiteLetter(binaryImg,mode=0):
    flag = True
    if mode==0:
        # 按面积比: 面积多的是背景
        count = 0
        for i, row in enumerate(binaryImg):
            for j, col in enumerate(row):
                if binaryImg[i][j]>200:
                    # 白色像素数量+1
                    count+=1
        if count>binaryImg.shape[0]*binaryImg.shape[1]*0.5:
            # 白色是背景
            flag = False
    elif mode==1:
        # 按边框
        count = 0
        for y in range(binaryImg.shape[0]):
            count += binaryImg[y][0]
            count += binaryImg[y][binaryImg.shape[1] - 1]
        for x in range(binaryImg.shape[1]):
            count += binaryImg[0][x]
            count += binaryImg[binaryImg.shape[0] - 1][x]
        if count > 0.2 * (binaryImg.shape[0] + binaryImg.shape[1]) * 255:
            flag = False
    else:
        print 'def isWhiteLetter(binaryImg,mode=0): mode erro!!'
    return flag

def expand(img, ratio=0.2):
    # img 黑字白底
    w = img.shape[1]
    h = img.shape[0]
    dy = (int)(h*ratio)
    expandImg = np.zeros((h+2*dy,w,3),np.uint8)
    expandImg[dy:dy+h,0:w] = img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    sample1 = [0,0,0]
    num1 = 0
    for x in range(w):
        if bw[0][x]>200:
            num1+=1
            sample1[0]+=img[0][x][0]
            sample1[1]+=img[0][x][1]
            sample1[2]+=img[0][x][2]
    if num1!=0:
        sample1[0] = sample1[0]/num1
        sample1[1] = sample1[1]/num1
        sample1[2] = sample1[2]/num1
    else:
        sample1=[0,0,0

                 ]
    for x in range(w):
        for y in range(dy):
            expandImg[y][x] = sample1
            # if bw[0][x]>200:
            #     expandImg[y][x] = img[0][x]
            # else:
            #     expandImg[y][x] = sample1

    sample1 = [0, 0, 0]
    num1 = 0
    for x in range(w):
        if bw[h-1][x] > 200:
            num1 += 1
            sample1[0] += img[h-1][x][0]
            sample1[1] += img[h-1][x][1]
            sample1[2] += img[h-1][x][2]
    if num1!=0:
        sample1[0] = sample1[0] / num1
        sample1[1] = sample1[1] / num1
        sample1[2] = sample1[2] / num1
    else:
        sample1=[0,0,0
                 ]
    for x in range(w):
        for y in range(dy):
            expandImg[h + dy + y][x] = sample1
            # if bw[h-1][x]>200:
            #     expandImg[h + dy + y][x] = img[h-1][x]
            # else:
            #     expandImg[h+dy+y][x] = sample1

    return expandImg


def textRectProcess(textImg):
    gray = cv2.cvtColor(textImg,cv2.COLOR_BGR2GRAY)
    _, bw= cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # 统一变为黑字白底
    iswhite = isWhiteLetter(bw,0)
    if iswhite:
        textImg = convert(textImg)
    # 扩充边界
    expandImg = expand(textImg)
    return expandImg

def getTextImgCrop(img,pts):
    # pts = [[(x1,y1),(x2,y1),(x2,y2),(x1,y2)] [] [] []] 三维list
    CropedImages = []
    for rect in pts:
        x1 = rect[0][0]
        y1 = rect[0][1]
        x2 = rect[2][0]
        y2 = rect[2][1]
        textImg = img[y1:y2,x1:x2]
        textRect = textRectProcess(textImg)
        CropedImages.append(textRect)

        cv2.imshow('img', textRect)
        # key = cv2.waitKey(0)
        # if key > 0:
        #     print '\n'
    # print CropedImages
    return CropedImages

def main_ctpn():
    imagePath = 'table/grid'
    resultPath = 'brand-results'
    savePath = 'brand-crop-result'

    # single test
    # imgname = 'WP0AA298XJS260479.jpg'
    # txt = 'res_'+ os.path.splitext(imgname)[0]+'.txt'
    # img = cv2.imread(os.path.join(imagePath,imgname))
    # pts = []
    # f = open(os.path.join(resultPath,'txt',txt))
    # while(True):
    #     line = f.readline() # str(min_x),str(min_y),str(max_x),str(max_y)
    #     line = line.strip('\r\n')
    #     if line==None or line=='':
    #          break
    #     coords = line.split(',',3)
    #     x1=int(coords[0]); y1=int(coords[1]);x2=int(coords[2]);y2=int(coords[3])
    #     p1=[x1,y1];p2=[x2,y1];p3=[x2,y2];p4=[x1,y2]
    #     rect = [p1,p2,p3,p4]
    #     pts.append(rect)
    # cropedImgs = getTextImgCrop(img, pts)

    # dir test
    for imgname in os.listdir(imagePath):
        try:
            txt = 'res_' + os.path.splitext(imgname)[0] + '.txt'
            img = cv2.imread(os.path.join(imagePath, imgname))
            pts = []
            f = open(os.path.join(resultPath, 'txt', txt))
            while (True):
                line = f.readline()  # str(min_x),str(min_y),str(max_x),str(max_y)
                line = line.strip('\r\n')
                if line == None or line == '':
                    break
                coords = line.split(',', 3)
                x1 = int(coords[0]);
                y1 = int(coords[1]);
                x2 = int(coords[2]);
                y2 = int(coords[3])
                p1 = [x1, y1];
                p2 = [x2, y1];
                p3 = [x2, y2];
                p4 = [x1, y2]
                rect = [p1, p2, p3, p4]
                pts.append(rect)
            cropedImgs = getTextImgCrop(img, pts)
        except:
            print '\n'


def find_lcseque(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    # print np.array(d)
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return ''.join(s)


if __name__=='__main__':
    string1 = '中国人民银行'
    string2 = '中国人们很行'
    print len(find_lcseque(string1, string2)) # 最长公共子序列
    print Levenshtein.distance(string1, string2)  # 编辑距离
    print Levenshtein.ratio(string1,string2) # 莱文斯坦比
    print Levenshtein.jaro(string1,string2) # jaro距离
    # main_ctpn()
    # main_line_restrain()