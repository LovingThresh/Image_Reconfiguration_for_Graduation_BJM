# -*- coding: utf-8 -*-
# @Time    : 2022/5/8 16:04
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm


# 目前需要处理的任务：
# 因为所有数据的文件名都是对应的，所以可以通过txt文件进行一一对应
# 1、图像超分任务：低分图像——高分图像
# 2、图像修复任务：缺损图像、缺损掩膜——修复图像

import os
import cv2
import albumentations as A
from torch.utils.data import Dataset
from torchvision import transforms

train_label_path = './bjm_data/raw_image/train'
val_label_path = './bjm_data/raw_image/valid'
test_label_path = './bjm_data/raw_image/test'


def get_txt_from_directory():
    for file_path, text in zip((train_label_path, val_label_path, test_label_path), ('train', 'valid', 'test')):
        files = os.listdir(file_path)
        with open('./bjm_data/{}.txt'.format(text), 'w') as f:
            for i in files[: -1]:
                f.write(i + '\n')
            f.write(files[-1])


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ColorJitter(p=0.2)
])


class Super_Resolution_Dataset(Dataset):
    def __init__(self, low_resolution_image_path, raw_image_path, data_txt, transformer=transform):
        self.raw_image_path = raw_image_path
        self.low_resolution_image_path = low_resolution_image_path
        self.data_txt = data_txt
        self.transform = transformer
        with open(self.data_txt, 'r') as f:
            self.file_list = f.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        self.raw_image = cv2.imread(os.path.join(self.raw_image_path, self.file_list[item]), cv2.COLOR_BGR2RGB)
        self.low_resolution_image = cv2.imread(os.path.join(self.low_resolution_image_path, self.file_list[item]), 
                                               cv2.COLOR_BGR2RGB)
        self.low_resolution_image = cv2.resize(self.low_resolution_image, (512, 512))
        if self.transform is None:
            pass

        else:
            self.transformed = self.transform(image=self.raw_image, mask=self.low_resolution_image)
            self.raw_image, self.low_resolution_image = self.transformed['image'], self.transformed['mask']
            self.raw_image, self.low_resolution_image = \
                transforms.ToTensor()(self.raw_image), transforms.ToTensor()(self.low_resolution_image)

        return self.raw_image, self.low_resolution_image


class Image_Reconstruction_Dataset(Dataset):
    def __init__(self, raw_image_path, defective_image_path, defective_mask_path, data_txt, transformer=None):
        self.raw_image_path = raw_image_path
        self.defective_image_path = defective_image_path
        self.defective_mask_path = defective_mask_path
        self.data_txt = data_txt
        self.transform = transformer
        with open(self.data_txt, 'r') as f:
            self.file_list = f.readlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        self.raw_image = cv2.imread(os.path.join(self.raw_image_path, self.file_list[item]), cv2.COLOR_BGR2RGB)
        self.defective_image = cv2.imread(os.path.join(self.defective_image_path, self.file_list[item]), cv2.COLOR_BGR2RGB)
        self.defective_mask = cv2.imread(os.path.join(self.defective_mask_path, self.file_list[item]), cv2.COLOR_BGR2RGB)
        
        if self.transform is None:
            pass

        else:
            self.raw_image, self.defective_image, self.defective_mask = self.transform(self.raw_image), \
                                                                   self.transform(self.defective_image), self.transform(self.defective_mask)

        return self.raw_image, self.defective_image, self.defective_mask
