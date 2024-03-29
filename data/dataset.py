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
from config import IMAGE_MEAN
from ctpn_utils import cal_rpn
import ipdb
import math


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
class VOCDataset(Dataset):
    def __init__(self,
                 datadir,
                 labelsdir):
        '''

        :param txtfile: image name list text file
        :param datadir: image's directory
        :param labelsdir: annotations' directory
        '''
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.datadir2="/home/like/data/VOC/VOC2007/JPEGImages2"
        # self.img_names = os.listdir(self.datadir)

        self.labelsdir = labelsdir
        # 取出文件夹中的文件
        self.img_names = []
        image_set_file = "/home/like/data/VOC/VOC2007/ImageSets/Main/trainval.txt"
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            self.img_names = [x.strip() + '.png' for x in f.readlines()]

        self.img_names2=[]
        image_set_file2 = "/home/like/data/VOC/VOC2007/ImageSets/Main/trainval2.txt"
        assert os.path.exists(image_set_file2), \
            'Path does not exist: {}'.format(image_set_file2)
        with open(image_set_file2) as f:
            self.img_names2 = [x.strip() + '.jpg' for x in f.readlines()]

        self.img_name=self.img_names+self.img_names2

        """实验"""
    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        #new work
        img_path = os.path.join(self.datadir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.datadir2, img_name)
        lim = img_name.split('_')[0]
        if lim not in ["img","image","tainchi"]:
            xml_path = os.path.join(self.labelsdir, img_name.replace('.png', '.xml'))
        else:
            self.labelsdir2 = "/home/like/data/VOC/VOC2007/Annotations2"
            xml_path = os.path.join(self.labelsdir2, img_name.replace('.jpg', '.xml'))
        img = cv2.imread(img_path)
        gtbox, _ = readxml(xml_path)
        h, w, c = img.shape
        if np.random.randint(2) == 1 and len(gtbox)>3:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], _ = cal_rpn((h, w), (math.ceil(h / 16), math.ceil(w / 16)), 16, gtbox)
        m_img = img - IMAGE_MEAN
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        cls = np.expand_dims(cls, axis=0)
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        """使用新/8的策略"""
        """得到新的gtbox"""
        newgtbox=[]
        for box in gtbox:
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            newgtbox.append([x1,y1,x1+8,y2])
            newgtbox.append([x1+8,y1,x2,y2])
        newgtbox=np.array(newgtbox)
        [cls_8,regr_8],_=cal_rpn((h, w), (math.ceil(h / 8), math.ceil(w / 8)), 8, newgtbox)
        regr_8 = np.hstack([cls_8.reshape(cls_8.shape[0], 1), regr_8])
        cls_8 = np.expand_dims(cls_8, axis=0)
        cls_8 = torch.from_numpy(cls_8).float()
        regr_8 = torch.from_numpy(regr_8).float()

        return m_img, cls, regr,cls_8,regr_8
