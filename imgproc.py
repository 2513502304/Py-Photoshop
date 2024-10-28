'''Pre-processing image module'''

import numpy as np
import cv2 as cv
import utils


def gray(image: np.ndarray) -> np.ndarray:
    '''灰度'''
    # 判断图像类型
    channels = utils.cheakImageType(image)
    # 灰度图
    if channels == 0:
        return image
    # 彩色图
    elif channels == 3 or channels == 4:
        # 灰度处理
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return grayImage


def monochrome(image: np.ndarray) -> np.ndarray:
    '''黑白'''
    # TODO
    return


def inversion(image: np.ndarray) -> np.ndarray:
    '''反相'''
    # 判断图像类型
    channels = utils.cheakImageType(image)
    # 灰度图或彩色图
    if channels == 0 or channels == 3 or channels == 4:
        # 反相处理
        inversionImage = cv.bitwise_not(image)
        return inversionImage


def thresholds(image: np.ndarray) -> np.ndarray:
    '''阈值'''
    # 判断图像类型
    channels = utils.cheakImageType(image)
    # 灰度图
    if channels == 0:
        # 二值化处理
        _, binaryImage = cv.threshold(image, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        return binaryImage
    # 彩色图
    elif channels == 3 or channels == 4:
        # 灰度处理
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 二值化处理
        _, binaryImage = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        return binaryImage


def toneHomogenization(image: np.ndarray) -> np.ndarray:
    '''色调均化'''
    # 判断图像类型
    channels = utils.cheakImageType(image)
    # 灰度图
    if channels == 0:
        # 色调均化处理
        toneHomogenizationImage = cv.equalizeHist(image)
        return toneHomogenizationImage
    # 彩色图
    elif channels == 3 or channels == 4:
        # 色调均化处理
        # 转换到 YCrCb 颜色空间
        ycrcbImage = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        # 将 Y 通道色调均匀
        channels = cv.split(ycrcbImage)
        cv.equalizeHist(channels[0], channels[0])
        # 将通道合成后转换回 BGR 颜色空间
        ycrcbImage = cv.merge(channels)
        toneHomogenizationImage = cv.cvtColor(ycrcbImage, cv.COLOR_YCrCb2BGR)
        return toneHomogenizationImage
