# -*- coding:utf-8 -*-
# '''
# Created on 18-12-27 上午10:34
#
# @Author: Greg Gao(laygin)
# '''

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
# from config import IMAGE_MEAN
# from ctpn_utils import cal_rpn
import ipdb
import math
# import config
def readxml(path):
    # 规整化成1080 X1920
    gtboxes = []
    imgfile = ''
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'filename' in elem.tag:
            imgfile = elem.text
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    xmin = float(attr.find('xmin').text)
                    ymin = float(attr.find('ymin').text)
                    xmax = float(attr.find('xmax').text)
                    ymax = float(attr.find('ymax').text)
                    #
                    # nx1=int(round((1920*xmin/width)))
                    # ny1=int(round((1080*ymin/height)))
                    # nx2=int(round((1920*xmax/width)))
                    # ny2=int(round((1080*ymax/height)))

                    gtboxes.append((xmin, ymin, xmax, ymax))

    return np.array(gtboxes), imgfile


# for ctpn text detection

def dataset():

    datadir = "/home/like/data/ctpn/VOCdevkit2007/VOC2007/JPEGImages"
    datadir2="/home/like/data/ctpn/VOCdevkit2007/VOC2007/JPEGImages2"
    # self.img_names = os.listdir(self.datadir)

    labelsdir = "/home/like/data/ctpn/VOCdevkit2007/VOC2007/Annotations"
    # 取出文件夹中的文件
    img_names = []
    image_set_file = "/home/like/data/ctpn/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt"
    assert os.path.exists(image_set_file), \
        'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        img_names = [x.strip() + '.jpg' for x in f.readlines()]

    img_names2=[]
    image_set_file2 = "/home/like/data/ctpn/VOCdevkit2007/VOC2007/ImageSets/Main/trainval2.txt"
    assert os.path.exists(image_set_file2), \
        'Path does not exist: {}'.format(image_set_file2)
    with open(image_set_file2) as f:
        img_names2 = [x.strip() + '.jpg' for x in f.readlines()]

    img_names=img_names+img_names2

    for idx in range(len(img_names)):
        img_name = img_names[idx]
        # print(img_name)
        #new work
        img_path = os.path.join(datadir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(datadir2, img_name)
        xml_path = ""
        lim = img_name.split('_')[0]
        if lim == "img":
            xml_path = os.path.join(labelsdir, img_name.replace('.jpg', '.xml'))
        else:
            labelsdir2 = "/home/like/data/ctpn/VOCdevkit2007/VOC2007/Annotations2"
            xml_path = os.path.join(labelsdir2, img_name.replace('.jpg', '.xml'))
        #new work end
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        gtbox, _ = readxml(xml_path)  # guiyihuatuxian
        if len(gtbox)<=2:
            print(img_path,len(gtbox))
            os.remove(img_path)
        # ipdb.set_trace()
        # # img=cv2.resize(img,(1080,1920),interpolation=cv2.INTER_CUBIC)
        # h, w, c = img.shape
        # # clip image
        # if np.random.randint(2) == 1 and len(gtbox)>3:
        #     img = img[:, ::-1, :]
        #     newx1 = w - gtbox[:, 2] - 1
        #     newx2 = w - gtbox[:, 0] - 1
        #     gtbox[:, 0] = newx1
        #     gtbox[:, 2] = newx2
        #
        # [cls, regr], _ = cal_rpn((h, w), (math.ceil(h / 16), math.ceil(w / 16)), 16, gtbox)
        #
        # m_img = img - IMAGE_MEAN
        #
        # regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        #
        # cls = np.expand_dims(cls, axis=0)
        #
        # # transform to torch tensor
        # m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        # cls = torch.from_numpy(cls).float()
        # regr = torch.from_numpy(regr).float()

dataset()