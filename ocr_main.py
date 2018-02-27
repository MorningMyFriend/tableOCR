# -*- coding:utf-8 -*-

import _init_paths
import preprocess
import cv2
import os
import re
import codecs
import numpy as np
import copy

from textdetection_hori.ctpn import ctpnport
from tesseract import tesseractport
# from textdetection_angle import ctpnport

this_dir = os.path.dirname(__file__)

#画框
def draw_boxes(img, boxes, color):
    for box in boxes:
        cv2.line(img, box[0], box[1], color, 2)
        cv2.line(img, box[1], box[2], color, 2)
        cv2.line(img, box[2], box[3], color, 2)
        cv2.line(img, box[3], box[0], color, 2)

def spiltChiWithEng(str1):
    utf8_str1 = str1.decode('utf-8')
    pat = re.compile(ur'[\u4e00-\u9fa5]')  # 这里是关键
    utf8_chi_str = ''
    for item in re.findall(pat, utf8_str1):  # 这里截取出中文字符
        utf8_chi_str = utf8_chi_str + item
    utf8_eng_str_list = pat.split(utf8_str1)
    utf8_eng_str = ''
    for elem in utf8_eng_str_list:
        for letter in elem:
            if letter.isalpha() or letter.isdigit():
                utf8_eng_str = utf8_eng_str + letter
    return utf8_chi_str, utf8_eng_str

#ctpn网络加载
ctpn_sess, ctpn_net = ctpnport.ctpnSource() #水平

#tesseract网络加载
tesseractport.tessInit()


if __name__=='__main__':
    #init pathes
    data_root_dir = os.path.join(this_dir, 'data') #数据路径
    raw_data_dir = os.path.join(data_root_dir, 'raw') #未处理数据路径
    res_data_dir = os.path.join(data_root_dir, 'res', 'shanghaibank') #结果路径
    edgeLine_dir = os.path.join(data_root_dir, 'edgeLine') #中间结果路径
    tmp_dir = os.path.join(this_dir, 'tmp') #临时文件夹
    res_txt = os.path.join(this_dir, 'data', 'tmp.txt')

    sub_raw_data_dirs = os.listdir(raw_data_dir)
    fp = codecs.open(res_txt, mode='a', encoding='utf-8')
    for sub_dir_name in sub_raw_data_dirs:
        if sub_dir_name != 'shanghaibank':
            continue
        sub_dir_path = os.path.join(raw_data_dir, sub_dir_name)
        img_names = os.listdir(sub_dir_path)
        for img_name in img_names:
            find_number_line = False
            img_path = os.path.join(sub_dir_path, img_name)
            try:
                image = cv2.imread(img_path)
                text_rects = ctpnport.getCharBlock(ctpn_sess, ctpn_net, image)

                numer_string = ''
                for box in text_rects:
                    minx = box[0][0] - 5
                    maxx = box[1][0] + 5
                    miny = box[0][1] - 3
                    maxy = box[3][1] + 3

                    #先判断box是否越界
                    if minx < 0: minx = 0
                    if maxx > image.shape[1] - 1: maxx = image.shape[1] - 1
                    if miny < 0: miny = 0
                    if maxy > image.shape[0] - 1: maxy = image.shape[0] - 1

                    if (maxx - minx) / (maxy - miny) < 10: #过滤寬高比例明显不符合的
                        continue

                    crop_img = image[miny : maxy, minx : maxx, :]
                    #cv2.imshow('crop_img', crop_img)
                    #cv2.waitKey(0)
                    result = tesseractport.tessRecognition(crop_img, tmp_dir)
                    result = result.decode('utf-8')

                    count_of_number = 0
                    orgnized_result = u'OCX流水号: '
                    numer_string = ''
                    for character in result:
                        if character.isdigit():
                            if count_of_number == 0 and character != u'2':
                                continue
                            count_of_number += 1
                            if count_of_number > 21:
                                break
                            orgnized_result += character
                            numer_string += character

                    if count_of_number > 10 and (result.find(u'流')!=-1 or result.find(u'水')!=-1
                                                 or result.find(u'OCX')!=-1 or result.find(u'2018')!=-1):
                        print img_name.decode('utf-8')
                        print 'orgnized_result:', orgnized_result
                        find_number_line = True
                        break

                if not find_number_line:
                    fp.writelines(img_name.decode('utf-8'))
                    fp.writelines('\n')
                else:
                    img_with_text = np.zeros([image.shape[0] + 40, image.shape[1], image.shape[2]], dtype=np.uint8)
                    img_with_text[40:, :] = copy.copy(image)
                    cv2.putText(img_with_text, numer_string, (25, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),2)
                    cv2.imwrite(os.path.join(res_data_dir, img_name), img_with_text)
                    cv2.imshow('img_with_text', img_with_text)
                    cv2.waitKey(0)
            except:
                print 'error!!!!'
    fp.close()