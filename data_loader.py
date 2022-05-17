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


def get_txt_from_directory(train_label_path, val_label_path, test_label_path):
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
    def __init__(self, low_resolution_image_path, raw_image_path, re_size,
                 data_txt, transformer=transform, down_scale=False):
        self.raw_image_path = raw_image_path
        self.low_resolution_image_path = low_resolution_image_path
        self.re_size = re_size
        self.data_txt = data_txt
        self.transform = transformer
        self.down_scale = down_scale
        with open(self.data_txt, 'r') as f:
            self.file_list = f.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        self.raw_image = cv2.imread(os.path.join(self.raw_image_path, self.file_list[item]))
        self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
        self.raw_image = cv2.resize(self.raw_image, self.re_size)
        self.low_resolution_image = cv2.imread(os.path.join(self.low_resolution_image_path, self.file_list[item]))
        self.low_resolution_image = cv2.cvtColor(self.low_resolution_image, cv2.COLOR_BGR2RGB)
        self.low_resolution_image = cv2.resize(self.low_resolution_image, self.re_size)
        if self.down_scale:
            self.low_resolution_image = cv2.resize(self.low_resolution_image, (int(self.re_size[0] / 2 ** self.down_scale),
                                                                               int(self.re_size[1] / 2 ** self.down_scale)))
        if self.transform is None:
            pass
        else:
            self.transformed = self.transform(image=self.raw_image, mask=self.low_resolution_image)
            self.raw_image, self.low_resolution_image = self.transformed['image'], self.transformed['mask']
            self.raw_image, self.low_resolution_image = \
                transforms.ToTensor()(self.raw_image), transforms.ToTensor()(self.low_resolution_image)

        return self.low_resolution_image, self.raw_image


class Image_Reconstruction_Dataset(Dataset):
    def __init__(self, defective_image_path, defective_mask_path, raw_image_path, data_txt, transformer=transform):
        self.raw_image_path = raw_image_path
        self.defective_image_path = defective_image_path
        self.defective_mask_path = defective_mask_path
        self.data_txt = data_txt
        self.transform = transformer
        with open(self.data_txt, 'r') as f:
            self.file_list = f.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        self.raw_image = cv2.imread(os.path.join(self.raw_image_path, self.file_list[item]))
        self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
        self.defective_image = cv2.imread(os.path.join(self.defective_image_path, self.file_list[item]))
        self.defective_image = cv2.cvtColor(self.defective_image, cv2.COLOR_BGR2RGB)
        self.defective_mask = cv2.imread(os.path.join(self.defective_mask_path, self.file_list[item]))
        self.defective_mask = cv2.cvtColor(self.defective_mask, cv2.COLOR_BGR2RGB)

        if self.transform is None:
            pass

        else:
            masks = [self.defective_image, self.defective_mask]
            self.transformed = self.transform(image=self.raw_image, masks=masks)
            self.raw_image = self.transformed['image']
            self.defective_image, self.defective_mask = self.transformed['masks']

            self.raw_image, self.defective_image, self.defective_mask = \
                transforms.ToTensor()(self.raw_image), transforms.ToTensor()(self.defective_image), \
                transforms.ToTensor()(self.defective_mask)

        return self.defective_image, self.defective_mask, self.raw_image


