#coding=utf-8
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


def lineDetect_main():
    # 直线检测 与 横条Grid 提取模块
    imgdir = 'data/image'
    resultdir = 'data'
    for img in os.listdir(imgdir):
        print 'image :',img
        imgPath = os.path.join(imgdir,img)
        image = cv2.imread(imgPath)
        w = image.shape[1]
        h = image.shape[0]
        maxlen = 1920.0
        if w > h:
            image = cv2.resize(image, ((int)(maxlen), (int)(maxlen / w * h)))
        else:
            image = cv2.resize(image, ((int)(maxlen / h * w), (int)(maxlen)))

        # 表格处理
        image, edges, veclines, horlines = tableprocess.findLine(image)
        if len(veclines)<2 or len(horlines)<2:
            print '未检测到表格线，退出！'
            exit(0)
        edges, gridImgs = tableprocess.getGrid(image, edges, veclines, horlines)
        for index,grid in enumerate(gridImgs):
            if grid == None:
                print 'grid None'
                continue
            xcoord = tableprocess.divideGrid(grid)
            cv2.imwrite(os.path.join(resultdir,'lineGrid', os.path.splitext(img)[0] + '-grid'+str(index)+'.jpg'), grid)

        cv2.imwrite(os.path.join(resultdir,'line',os.path.splitext(img)[0] + '-line.jpg'), edges)


def ctpnGrid_main():
    # gridImgs , xcoords, reultPath
    gridImgPath = 'data/lineGrid'
    for img in os.listdir(gridImgPath):
        gridImg = cv2.imread(img)
        text_rects = ctpnport.getCharBlock(ctpn_sess, ctpn_net, gridImg)
        for box in text_rects:
            minx = box[0][0]
            maxx = box[1][0]
            miny = box[0][1]
            maxy = box[3][1]

if __name__ == '__main__':
    lineDetect_main()