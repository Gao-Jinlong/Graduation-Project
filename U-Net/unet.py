import colorsys
import copy
import time

import cv2
import numpy as np
import PIL import Image

from nets.unet import Unet as unet

#------------------------#
#        Unet类
#------------------------#
class Unet(object):
    _defaults = {
    #-----------------------------------------------------------------------#
    #  model_path 指向logs文件夹下的全职文件
    #  训练好后logs文件夹下存在多个权值文件，选择验证损失及较低的即可。
    #  验证损失集较低不代表mIOU较高，仅代表该权值在验证集上的泛化性能较好。
    #
    #  mIOU：计算真实值和预测值两个集合的交集和并集之比。
    #        这个比例可以变形为TP（交集）比上TP、FP、FN之和（并集）
    #  mIOU（Mean Intersection over Union，MIoU 均交幷比）
    #-----------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_voc.h5',
    #  要区分的类的个数+1
    "num_classes"   : 21,
    #  输入图片大小
    "input_shape"   : [512, 512],
    #  识别结果与原图混合
    "blend"         : True
    }

    #-----------------------------------------------------------------------#
    #  初始化Unet
    #-----------------------------------------------------------------------#
    def  __init__(self, **kwargs):
        self.__dict__.update(self._defaults)    #  更新字典
        for name, value in kwargs.items():
