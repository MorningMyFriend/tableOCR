# -*- coding:utf-8 -*-

import _init_paths
import preprocess
import tableprocess
import cv2
import os
import re
import codecs
import numpy as np
import copy

from textdetection_hori.ctpn import ctpnport
from tesseract import tesseractport

this_dir = os.path.dirname(__file__)

#ctpn网络加载
ctpn_sess, ctpn_net = ctpnport.ctpnSource() #水平

#tesseract网络加载
tesseractport.tessInit()

def lineDetect(imgdir, resultdir):
    # 直线检测 与 横条Grid 提取模块; 返回彩色横条grid和竖向分割坐标，保存中间结果
    # imgdir = 'data/image'
    # resultdir = 'data'
    for img in os.listdir(imgdir):
        print 'image :', img
        imgPath = os.path.join(imgdir, img)
        image = cv2.imread(imgPath)
        # image = tableprocess.enhanceImg(image)
        cv2.imwrite('/home/wurui/project/tableOCR/data/image/enhance.png',image)
        # cv2.imshow('red',image)
        # key = cv2.waitKey(0)
        # if key>0:
        #     print ' '
        w = image.shape[1]
        h = image.shape[0]
        maxlen = 1920.0
        if w > h:
            image = cv2.resize(image, ((int)(maxlen), (int)(maxlen / w * h)))
        else:
            image = cv2.resize(image, ((int)(maxlen / h * w), (int)(maxlen)))

        # 表格处理
        image, edges, veclines, horlines = tableprocess.findLine(image)
        if len(veclines) < 2 or len(horlines) < 2:
            print '未检测到表格线，退出！'
            exit(0)
        edges, gridImgs = tableprocess.getGrid(image, edges, veclines, horlines)
        xcoords = []
        for index, grid in enumerate(gridImgs):
            if grid == None:
                print 'grid None'
                continue
            # 竖向分割坐标
            xcoord = tableprocess.divideGrid(grid)
            xcoords.append(xcoord)
            cv2.imwrite(os.path.join(resultdir, 'lineGrid', os.path.splitext(img)[0] + '-grid' + str(index) + '.jpg'),
                        grid)

        cv2.imwrite(os.path.join(resultdir, 'line', os.path.splitext(img)[0] + '-line.jpg'), edges)
        return gridImgs,xcoords


def singleGrid(gridImgs, xcoords, resultPath):
    ctpngrids = []
    if len(gridImgs) != len(xcoords):
        print 'length gridImg != length xcoords: warning!!!!!!!!!!!!!!!!!!!!!!!!!!'
    for lineIndex, gridImg in enumerate(gridImgs):
        linegrid = []
        # cv2.imshow('grid line',gridImg)
        xcoord = xcoords[lineIndex]
        if xcoord == None:
            xcoord = []
        w = gridImg.shape[1]
        h = gridImg.shape[0]
        xcoord.append(w)
        xcoord.insert(0,0)
        for i in range(0,len(xcoord)-1):
            if xcoord[i+1]-xcoord[i] < 10:
                continue
            img = np.zeros((xcoord[i+1]-xcoord[i],h),np.uint8)
            img = gridImg[0:h,xcoord[i]:xcoord[i+1]]
            # cv2.imshow('grid',img)
            # key = cv2.waitKey(0)
            # if key>0:
            #     print ' '
            cv2.imwrite(os.path.join(resultPath, str(lineIndex) + '-'+str(i)+'.jpg'), img)
            linegrid.append(img)
        ctpngrids.append(linegrid)

    return ctpngrids


def ctpnGrid(gridImgs, xcoords, reultPath):
    ctpngrids = []
    if len(gridImgs)!=len(xcoords):
        print 'length gridImg != length xcoords: warning!!!!!!!!!!!!!!!!!!!!!!!!!!'
    for lineIndex,gridImg in enumerate(gridImgs):
        xcoord = xcoords[lineIndex]
        if xcoord == None:
            xcoord = []
        w = gridImg.shape[1]
        h = gridImg.shape[0]
        text_rects = ctpnport.getCharBlock(ctpn_sess, ctpn_net, gridImg)
        linegrid = []
        gridIndex = 0
        # ctpn检测的box中，用xcoords的分界线划分成真正的单元格
        text_rects.sort()

        # # debug
        tmpGridImg = copy.copy(gridImg)
        for box in text_rects:
            minx = max(box[0][0] - 5, 0)
            maxx = min(box[1][0] + 5, w)
            miny = max(box[0][1] - 3, 0)
            maxy = min(box[3][1] + 3, h)
            cv2.rectangle(tmpGridImg,(minx,miny),(maxx,maxy),(0,255,0),2)
        cv2.imshow('gridImg',tmpGridImg)
        key = cv2.waitKey(0)
        if key>0:
            print ' '

        for box in text_rects:
            minx = max(box[0][0]-5,0)
            maxx = min(box[1][0]+5,w)
            miny = max(box[0][1]-3,0)
            maxy = min(box[3][1]+3,h)
            xgrid = []
            xgrid.append(minx)
            xgrid.append(maxx)
            for x in xcoord:
                if x > minx and x < maxx:
                    xgrid.append(x)
            xgrid.sort()
            for i in range(len(xgrid)-1):
                x1 = xgrid[i]
                x2 = xgrid[i+1]
                if x2-x1 < 10:
                    continue
                grid = np.zeros((x2 - x1, maxy - miny), np.uint8)
                grid = gridImg[miny:maxy, x1:x2]
                # 扩充边界
                grid = preprocess.expand(grid,ratio=0.2)
                linegrid.append(grid)
                gridIndex+=1
                # # debug
                # cv2.imshow('grid',grid)
                # key = cv2.waitKey(0)
                # if key>0:
                #     cv2.imwrite(os.path.join(reultPath, str(lineIndex) + str(gridIndex)+'.jpg'), grid)
                #     continue
        ctpngrids.append(linegrid)
    return ctpngrids


if __name__=='__main__':
    #init pathes
    data_root_dir = os.path.join(this_dir, 'data') #数据路径
    tmp_dir = os.path.join(this_dir, 'temp') #临时文件夹
    table_txt = os.path.join(this_dir, 'data','result.txt')

    # 预处理获得条形grid和竖向分界坐标
    gridImages, xcoords = lineDetect(imgdir=os.path.join(data_root_dir,'image'),
                                     resultdir=data_root_dir)

    # ctpngrids = {[grid1_1 grid1_2...] [grid2_1...]...} 得到白底黑字BGR单元格图像
    ctpngrids = singleGrid(gridImages, xcoords, os.path.join(data_root_dir,'ctpnGrid'))
    # ctpngrids = ctpnGrid(gridImages, xcoords, os.path.join(data_root_dir,'ctpnGrid'))

    # # tesseract
    # f = codecs.open(table_txt,'wb','utf-8')
    # for line in ctpngrids:
    #     for grid in line:
    #         result = tesseractport.tessRecognition(grid, tmp_dir)
    #         result = "".join(result.split())
    #         result = result.decode('utf-8')
    #         print result
    #         f.write('| ' + result + ' |')
            # print result
            # if result=='':
            #     f.write('|    |')
            # elif result[-1:] == '\n':
            #     result = result[0:-1]
            #     f.write('|    '+result+'|')
            # else:
            #     f.write('|    ' + result + '|')


            # print result
            # cv2.imshow('grid',grid)
            # key = cv2.waitKey(0)
            # if key>0:
            #     continue
        # f.write('\n')

