# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 23:09
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : visualize.py
# @Software: PyCharm
import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt


def dim_to_numpy(x: torch.Tensor or np.array):
    if x.ndim == 4 & torch.is_tensor(x):
        x = x.squeeze(0)
        x = x.cpu().numpy()
        x = x.transpose(1, 2, 0)
    elif x.ndim == 3 & torch.is_tensor(x):
        x = x.cpu().numpy()
    return x


def plot(x: np.array or torch.Tensor, size=(10, 10)):
    x = dim_to_numpy(x)
    assert x.ndim == 3 or (x.ndim == 2)
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(x)
    plt.show()


def visualize_model(model: torch.nn.Module, image, image_pair=False):
    model.eval()
    assert image.ndim == 4
    prediction = model(image)
    if image_pair:
        image, prediction = dim_to_numpy(image), dim_to_numpy(prediction)
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(prediction)
        plt.show()
    else:
        plot(prediction)
