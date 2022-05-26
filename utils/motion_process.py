# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 19:30
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : motion_process.py
# @Software: PyCharm

# Inference : https://blog.csdn.net/googler_offer/article/details/88841048

import numpy.fft as fft
import numpy as np
import math
import cv2


def motion_process(image_size, motion_angle, motion_dis):
    psf = np.zeros(image_size)
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2

    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)

    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        psf[int(x_center - x_offset), int(y_center + y_offset)] = 1

    return psf / psf.sum()


def make_blurred(input, psf, eps):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(psf) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def wiener(input, psf, eps, K=0.01):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(psf) + eps
    # np.conj是计算共轭值
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


def inverse(input, psf, eps):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(psf) + eps
    result = fft.ifft2(input_fft / PSF_fft)
    result = np.abs(fft.fftshift(result))
    return result


def de_focus(input, degree=21):
    blurred = cv2.GaussianBlur(input, ksize=(degree, degree), sigmaX=0, sigmaY=0)
    return blurred


PSF = motion_process((448, 448), 60, 5)
image = cv2.imread('L:/ALASegmentationNets_v2/Data/Stage_4/test/img/CFD_005.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

blur_image = de_focus(image)
cv2.imshow("blur_image", blur_image)
cv2.waitKey(0)

cv2.imshow('blurred_image', np.uint8(image))
cv2.waitKey(0)
R, G, B = cv2.split(image)
blurred_image_R, blurred_image_G, blurred_image_B = np.abs(make_blurred(R, PSF, 1e-3)), \
                                                    np.abs(make_blurred(G, PSF, 1e-3)), \
                                                    np.abs(make_blurred(B, PSF, 1e-3))
blurred_image = cv2.merge([blurred_image_R, blurred_image_G, blurred_image_B])
cv2.imshow('blurred_image', np.uint8(blurred_image))
cv2.waitKey(0)


R, G, B = cv2.split(blurred_image)
blurred_image_R, blurred_image_G, blurred_image_B = np.abs(wiener(R, PSF, 1e-3)), \
                                                    np.abs(wiener(G, PSF, 1e-3)), \
                                                    np.abs(wiener(B, PSF, 1e-3))
blurred_image = cv2.merge([blurred_image_R, blurred_image_G, blurred_image_B])
cv2.imshow('blurred_image', np.uint8(blurred_image))
cv2.waitKey(0)


R, G, B = cv2.split(blurred_image)
blurred_image_R, blurred_image_G, blurred_image_B = np.abs(inverse(R, PSF, 1e-3)), \
                                                    np.abs(inverse(G, PSF, 1e-3)), \
                                                    np.abs(inverse(B, PSF, 1e-3))
blurred_image = cv2.merge([blurred_image_R, blurred_image_G, blurred_image_B])

cv2.imshow('blurred_image', np.uint8(blurred_image))
cv2.waitKey(0)


