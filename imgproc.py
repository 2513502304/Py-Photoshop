"""Pre-processing image module"""

import cv2 as cv
import numpy as np

import utils

# ==================================== 图像 ------------------------------------


# ------------------------------------ 模式 ------------------------------------
def gray(image: np.ndarray) -> np.ndarray:
    """
    灰度

    Args:
        image (np.ndarray): 输入图像

    Returns:
        np.ndarray: 灰度图像
    """
    # 判断图像类型
    channels = utils.cheakImageType(image)
    # 灰度图
    if channels == 0:
        return image.copy()
    # 彩色图
    elif channels == 3:
        # 灰度处理
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return grayImage
    # 带有 Alpha 通道的彩色图
    elif channels == 4:
        # 灰度处理
        grayImage = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
        return grayImage


# ------------------------------------ 调整 ------------------------------------
def monochrome(image: np.ndarray) -> np.ndarray:
    """
    黑白

    Args:
        image (np.ndarray): 输入图像

    Returns:
        np.ndarray: 黑白图像
    """
    # 判断图像类型
    channels = utils.cheakImageType(image)
    # 灰度图
    if channels == 0:
        return image.copy()
    # 彩色图
    elif channels == 3:
        # 灰度处理
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return grayImage
    # 带有 Alpha 通道的彩色图
    elif channels == 4:
        # 灰度处理
        grayImage = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
        return grayImage


def inversion(image: np.ndarray) -> np.ndarray:
    """
    反相

    Args:
        image (np.ndarray): 输入图像

    Returns:
        np.ndarray: 反相图像
    """
    # 判断图像类型
    channels = utils.cheakImageType(image)
    # 灰度图或彩色图或带有 Alpha 通道的彩色图
    if channels == 0 or channels == 3 or channels == 4:
        # 反相处理
        inversionImage = cv.bitwise_not(image)
        return inversionImage


def thresholds(image: np.ndarray, thresh: int | None = None) -> np.ndarray:
    """
    阈值

    Args:
        image (np.ndarray): 输入图像
        thresh (int | None, optional): 输入图像阈值，若为 None，则自动阈值. Defaults to None.

    Returns:
        np.ndarray: thresh
    """
    # 判断图像类型
    channels = utils.cheakImageType(image)
    # 灰度图
    if channels == 0:
        # 二值化处理
        _, binaryImage = cv.threshold(image, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) if thresh is None else cv.threshold(image, thresh, 255, cv.THRESH_BINARY)
        return binaryImage
    # 彩色图
    elif channels == 3:
        # 灰度处理
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 二值化处理
        _, binaryImage = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) if thresh is None else cv.threshold(image, thresh, 255, cv.THRESH_BINARY)
        return binaryImage
    # 带有 Alpha 通道的彩色图
    elif channels == 4:
        # 灰度处理
        grayImage = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
        # 二值化处理
        _, binaryImage = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) if thresh is None else cv.threshold(image, thresh, 255, cv.THRESH_BINARY)
        return binaryImage


def toneHomogenization(image: np.ndarray) -> np.ndarray:
    """
    色调均化

    Args:
        image (np.ndarray): 输入图像

    Returns:
        np.ndarray: 色调均化图像
    """
    # 判断图像类型
    channels = utils.cheakImageType(image)
    # 创建 CLAHE 对象
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 灰度图
    if channels == 0:
        # 色调均化处理
        # toneHomogenizationImage = cv.equalizeHist(image)
        toneHomogenizationImage = clahe.apply(image)
        return toneHomogenizationImage
    # 彩色图
    elif channels == 3:
        # 转换到 YCrCb 颜色空间
        ycrcbImage = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        # 将 Y 通道色调均匀
        # ycrcbImage[:, :, 0] = cv.equalizeHist(ycrcbImage[:, :, 0])
        ycrcbImage[:, :, 0] = clahe.apply(ycrcbImage[:, :, 0])
        # 转换回 BGR 颜色空间
        toneHomogenizationImage = cv.cvtColor(ycrcbImage, cv.COLOR_YCrCb2BGR)
        return toneHomogenizationImage
    # 带有 Alpha 通道的彩色图
    elif channels == 4:
        # 单独分离 Alpha 通道
        a = image[:, :, 3]
        # 去除 Alpha 通道
        bgr = image[:, :, :3]
        # 转换到 YCrCb 颜色空间
        ycrcbImage = cv.cvtColor(bgr, cv.COLOR_BGR2YCrCb)
        # 将 Y 通道色调均匀
        # ycrcbImage[:, :, 0] = cv.equalizeHist(ycrcbImage[:, :, 0])
        ycrcbImage[:, :, 0] = clahe.apply(ycrcbImage[:, :, 0])
        # 转换回 BGR 颜色空间
        toneHomogenizationImage = cv.cvtColor(ycrcbImage, cv.COLOR_YCrCb2BGR)
        # 合并 Alpha 通道
        toneHomogenizationImage = cv.merge([toneHomogenizationImage, a])
        return toneHomogenizationImage


# ==================================== 滤镜 ====================================


# ------------------------------------ 像素化 ------------------------------------
def mosaic(image: np.ndarray, n: int = 8) -> np.ndarray:
    """
    马赛克

    Args:
        image (np.ndarray): 输入图像
        n (int, optional): 像素化大小. Defaults to 8.

    Returns:
        np.ndarray: 马赛克图像
    """
    # 缩小图像，即降采样：使用像素面积关系进行重采样。它可能是图像降采样的首选方法，因为它能产生无摩尔纹的结果。但是当图像被放大时，它与 INTER_NEAREST 方法相似
    image = cv.resize(image, (image.shape[1] // n, image.shape[0] // n), interpolation=cv.INTER_AREA)
    # 放大图像，即升采样：使用最近邻插值方法。它选择最接近目标像素位置的原始像素值作为调整后像素的值。这种方法计算速度快，但可能会导致图像边缘的锯齿状效果
    image = cv.resize(image, (image.shape[1] * n, image.shape[0] * n), interpolation=cv.INTER_NEAREST)
    return image


"""
[混合模式:Adobe](https://helpx.adobe.com/cn/photoshop/using/blending-modes.html)
[混合模式:维基百科](https://en.wikipedia.org/wiki/Blend_modes)
Normal: 普通
    - Normal: 正常  
        - 编辑或绘制每个像素，使其成为结果色。这是默认模式。（在处理位图图像或索引颜色图像时，“正常”模式也称为阈值。）
    - Dissolve: 溶解
        - 编辑或绘制每个像素，使其成为结果色。但是，根据任何像素位置的不透明度，结果色由基色或混合色的像素随机替换。
Darken: 变暗
    - Darken: 变暗
    - Multiply: 正片叠底
    - Color Burn: 颜色加深
    - Linear Burn: 线性加深
    - Darker Color: 深色
Lighten: 变亮
    - Lighten: 变亮
    - Screen: 滤色
    - Color Dodge: 颜色减淡
    - Linear Dodge (Add): 线性减淡（添加）
    - Lighter Color: 浅色
Contrast: 对比
    - Overlay: 叠加
    - Soft Light: 柔光
    - Hard Light: 强光
    - Vivid Light: 亮光
    - Linear Light: 线性光
    - Pin Light: 点光
    - Hard Mix: 实色混合
Inversion: 差集
    - Difference: 差值
    - Exclusion: 排除
Cancellation: 抵消
    - Subtract: 减去
    - Divide: 划分
Component: 组件
    - Hue: 色相
    - Saturation: 饱和度
    - Color: 颜色
    - Luminosity: 明度
"""


# ------------------------------------ NORMAL GROUP ------------------------------------
def normal(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    正常
    
    Note:
        正常效果
        f(a, b) = a

    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 正常处理
    return blend.copy()


def dissolve(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    溶解

    Note:
        溶解效果
        f(a, b) = random(a, b)
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 单独分离 Alpha 通道
    blend_a = blend_color[:, :, 3]
    # 获取溶解像素比例
    ratio = blend_a.mean() / 255
    # 结果图像
    result = base_color.copy()
    # 生成随机掩码
    np.random.seed(0)
    mask = np.random.rand(*blend_a.shape) < ratio
    # 溶解处理
    result[mask] = blend_color[mask]
    return result


# ------------------------------------ DARKEN GROUP ------------------------------------
def darken(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    变暗

    Note:
        变暗效果
        f(a, b) = min(a, b)
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 变暗处理
    darkenImage = cv.min(blend_color, base_color)
    return darkenImage


def multiply(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    正片叠底

    Note:
        正片叠底效果
        f(a, b) = a * b
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 正片叠底处理
    multiplyImage = cv.multiply(blend_color, base_color, scale=1 / 255, dtype=cv.CV_8U)
    return multiplyImage


def colorBurn(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    颜色加深

    Note:
        颜色加深效果
        f(a, b) = 1 - (1 - b) / a = (a + b - 1) / a
        clip negative values to zero, i.e. where (1 - b) > a, (1 - b) = a, or (a + b - 1) < 0, (a + b - 1) = 0
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 颜色加深处理
    colorBurnImage = 255 - cv.divide((255 - base_color), blend_color, scale=255, dtype=cv.CV_8U)
    return colorBurnImage


def linearBurn(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    线性加深

    Note:
        线性加深效果
        f(a, b) = a + b - 1
        clip negative values to zero, i.e. where a + b - 1 < 0, a + b - 1 = 0
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend).astype(np.float32)  # 防止溢出
    base_color = utils.universalColor(base).astype(np.float32)  # 防止溢出
    # 线性加深处理
    linearBurnImage = np.where(blend_color + base_color - 255 >= 0, blend_color + base_color - 255, 0)
    return linearBurnImage.astype(np.uint8)


def darkerColor(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    深色

    Note:
        深色效果
        if a.r + a.g + a.b < b.r + b.g + b.b:
            f(a, b) = a
        elif a.r + a.g + a.b >= b.r + b.g + b.b:
            f(a, b) = b
            
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 结果图像
    result = np.zeros_like(blend_color, dtype=np.uint8)
    # 计算颜色和
    blend_sum = np.sum(blend_color, axis=-1)
    base_sum = np.sum(base_color, axis=-1)
    # 掩码图像
    mask = blend_sum < base_sum
    # 深色处理
    result[mask] = blend_color[mask]
    result[~mask] = base_color[~mask]
    return result


# ------------------------------------ LIGHTEN GROUP ------------------------------------
def lighten(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    变亮

    Note:
        变亮效果
        f(a, b) = max(a, b)
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 变亮处理
    lightenImage = cv.max(blend_color, base_color)
    return lightenImage


def screen(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    滤色

    Note:
        滤色效果
        f(a, b) = 1 - (1 - a) * (1 - b)
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 滤色处理
    screenImage = 255 - cv.multiply(255 - blend_color, 255 - base_color, scale=1 / 255, dtype=cv.CV_8U)
    return screenImage


def colorDodge(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    颜色减淡

    Note:
        颜色减淡效果
        f(a, b) = b / (1 - a)
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 颜色减淡处理
    colorDodgeImage = cv.divide(base_color, 255 - blend_color, scale=255, dtype=cv.CV_8U)
    return colorDodgeImage


def linearDodge(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    线性减淡（添加）

    Note:
        线性减淡效果
        f(a, b) = a + b
        clip values to 255, i.e. where a + b > 255, a + b = 255
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 线性减淡处理
    linearDodgeImage = cv.add(blend_color, base_color, dtype=cv.CV_8U)
    return linearDodgeImage


def lighterColor(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    浅色

    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Note:
        浅色效果
        if a.r + a.g + a.b > b.r + b.g + b.b:
            f(a, b) = a
        elif a.r + a.g + a.b <= b.r + b.g + b.b:
            f(a, b) = b
            
    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 结果图像
    result = np.zeros_like(blend_color, dtype=np.uint8)
    # 计算颜色和
    blend_sum = np.sum(blend_color, axis=-1)
    base_sum = np.sum(base_color, axis=-1)
    # 掩码图像
    mask = blend_sum > base_sum
    # 浅色处理
    result[mask] = blend_color[mask]
    result[~mask] = base_color[~mask]
    return result


# ------------------------------------ CONTRAST GROUP ------------------------------------
def overlay(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    叠加

    Note:
        叠加效果
        if b < 0.5:
            f(a, b) = 2 * a * b
        elif b >= 0.5:
            f(a, b) = 1 - 2 * (1 - a) * (1 - b)
            
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 结果图像
    result = np.zeros_like(blend_color, dtype=np.uint8)
    # 掩码图像
    mask = base_color < 128
    # 叠加处理
    result[mask] = cv.multiply(blend_color[mask], base_color[mask], scale=2 / 255, dtype=cv.CV_8U).reshape(-1)
    result[~mask] = 255 - cv.multiply(255 - blend_color[~mask], 255 - base_color[~mask], scale=2 / 255, dtype=cv.CV_8U).reshape(-1)
    return result


def softLight(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    柔光

    Note:
        柔光效果
        if a < 0.5:
            f(a, b) = 2 * a * b + b^2 * (1 - 2 * a)
        elif a >= 0.5:
            f(a, b) = 2 * b * (1 - a) + sqrt(b) * (2 * a - 1)
            
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend).astype(np.float32)  # 高精度计算
    base_color = utils.universalColor(base).astype(np.float32)  # 高精度计算
    # 结果图像
    result = np.zeros_like(blend_color, dtype=np.uint8)
    # 掩码图像
    mask = blend_color < 128
    # 柔光处理
    result[mask] = 2 * blend_color[mask] * base_color[mask] / 255 + base_color[mask]**2 / 255 * (1 - 2 * blend_color[mask] / 255)
    result[~mask] = 2 * base_color[~mask] * (255 - blend_color[~mask]) / 255 + np.sqrt(base_color[~mask] / 255) * (2 * blend_color[~mask] - 255)
    return result


def hardLight(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    强光
    
    Note:
        强光效果
        if a < 0.5:
            f(a, b) = 2 * a * b
        elif a >= 0.5:
            f(a, b) = 1 - 2 * (1 - a) * (1 - b)

    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 结果图像
    result = np.zeros_like(blend_color, dtype=np.uint8)
    # 掩码图像
    mask = blend_color < 128
    # 强光处理
    result[mask] = cv.multiply(blend_color[mask], base_color[mask], scale=2 / 255, dtype=cv.CV_8U).reshape(-1)
    result[~mask] = 255 - cv.multiply(255 - blend_color[~mask], 255 - base_color[~mask], scale=2 / 255, dtype=cv.CV_8U).reshape(-1)
    return result


def vividLight(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    亮光
    
    Note:
        亮光效果
        if a < 0.5:
            f(a, b) = 1 - (1 - b) / (2 * a)
        elif a >= 0.5:
            f(a, b) = b / (2 * (1 - a))

    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 结果图像
    result = np.zeros_like(blend_color, dtype=np.uint8)
    # 掩码图像
    mask = blend_color < 128
    # 亮光处理
    result[mask] = 255 - cv.divide(255 - base_color[mask], blend_color[mask], scale=255 / 2, dtype=cv.CV_8U).reshape(-1)
    result[~mask] = cv.divide(base_color[~mask], 255 - blend_color[~mask], scale=255 / 2, dtype=cv.CV_8U).reshape(-1)
    return result


def linearLight(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    线性光

    Note:
        线性光效果
        f(a, b) = b + 2 * a - 1
        clip negative values to zero, i.e. where b + 2 * a - 1 < 0, b + 2 * a - 1 = 0
        clip values to 255, i.e. where b + 2 * a - 1 > 255, b + 2 * a - 1 = 255
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: _description_
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend).astype(np.float32)  # 防止溢出
    base_color = utils.universalColor(base).astype(np.float32)  # 防止溢出
    # 线性光处理
    linearLightImage = base_color + 2 * blend_color - 255
    return linearLightImage.clip(0, 255).astype(np.uint8)


def pinLight(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    点光

    Note:
        点光效果
        if b < 2 * a - 1:
            f(a, b) = 2 * a - 1
        elif 2 * a - 1 <= b <= 2 * a:
            f(a, b) = b
        elif b > 2 * a:
            f(a, b) = 2 * a
            
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend).astype(np.float32)  # 防止溢出
    base_color = utils.universalColor(base).astype(np.float32)  # 防止溢出
    # 结果图像
    result = np.zeros_like(blend_color, dtype=np.uint8)
    # 掩码图像
    mask1 = base_color < 2 * blend_color - 255
    mask2 = base_color > 2 * blend_color
    mask3 = ~mask1 & ~mask2
    # 点光处理
    result[mask1] = 2 * blend_color[mask1] - 255
    result[mask2] = 2 * blend_color[mask2]
    result[mask3] = base_color[mask3]
    return result


def hardMix(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    实色混合

    Note:
        实色混合效果
        if a < 1 - b:
            f(a, b) = 0
        elif a >= 1 - b:
            f(a, b) = 1
            
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 结果图像
    result = np.zeros_like(blend_color, dtype=np.uint8)
    # 掩码图像
    mask = blend_color < 255 - base_color
    # 实色混合处理
    result[mask] = 0
    result[~mask] = 255
    return result


# ------------------------------------ INVERSION GROUP ------------------------------------
def difference(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    差值

    Note:
        差值效果
        f(a, b) = |b - a|
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 差值处理
    differenceImage = cv.absdiff(base_color, blend_color)
    return differenceImage


def exclusion(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    排除

    Note:
        排除效果
        f(a, b) = a + b - 2 * a * b
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend).astype(np.float32)  # 防止溢出
    base_color = utils.universalColor(base).astype(np.float32)  # 防止溢出
    # 排除处理
    exclusionImage = blend_color + base_color - 2 * blend_color * base_color / 255
    return exclusionImage.astype(np.uint8)


# ------------------------------------ CANCELLATION GROUP ------------------------------------
def subtract(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    减去

    Note:
        减去效果
        f(a, b) = b - a
        clip negative values to zero, i.e. where b - a < 0, b - a = 0
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 减去处理
    subtractImage = cv.subtract(base_color, blend_color)
    return subtractImage


def divide(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    划分

    Note:
        划分效果
        f(a, b) = b / a
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 划分处理
    divideImage = cv.divide(base_color, blend_color)
    return divideImage


# ------------------------------------ COMPONENT GROUP ------------------------------------
def hue(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    色相

    Note:
        色相效果
        c.h, c.s, c.v = a.h, b.s, b.v
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 拆分 Alpha 通道
    blend_a = blend_color[:, :, 3]
    base_a = base_color[:, :, 3]
    # 混合 Alpha 通道
    result_a = cv.addWeighted(blend_a, 0.5, base_a, 0.5, 0)
    # 拆分 BGR 通道
    blend_bgr = blend_color[:, :, :3]
    base_bgr = base_color[:, :, :3]
    # 转换到 HSV 颜色空间
    blend_hsv = cv.cvtColor(blend_bgr, cv.COLOR_BGR2HSV)
    base_hsv = cv.cvtColor(base_bgr, cv.COLOR_BGR2HSV)
    # 色相处理
    result_hsv = np.zeros_like(blend_hsv, dtype=np.uint8)
    result_hsv[:, :, 0] = blend_hsv[:, :, 0]  # h
    result_hsv[:, :, 1] = base_hsv[:, :, 1]  # s
    result_hsv[:, :, 2] = base_hsv[:, :, 2]  # v
    # 转换回 BGR 颜色空间
    result_bgr = cv.cvtColor(result_hsv, cv.COLOR_HSV2BGR)
    # 合并 Alpha 通道
    result = cv.merge([result_bgr, result_a])
    return result


def saturation(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    饱和度

    Note:
        饱和度效果
        c.h, c.s, c.v = b.h, a.s, b.v
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 拆分 Alpha 通道
    blend_a = blend_color[:, :, 3]
    base_a = base_color[:, :, 3]
    # 混合 Alpha 通道
    result_a = cv.addWeighted(blend_a, 0.5, base_a, 0.5, 0)
    # 拆分 BGR 通道
    blend_bgr = blend_color[:, :, :3]
    base_bgr = base_color[:, :, :3]
    # 转换到 HSV 颜色空间
    blend_hsv = cv.cvtColor(blend_bgr, cv.COLOR_BGR2HSV)
    base_hsv = cv.cvtColor(base_bgr, cv.COLOR_BGR2HSV)
    # 饱和度处理
    result_hsv = np.zeros_like(blend_hsv, dtype=np.uint8)
    result_hsv[:, :, 0] = base_hsv[:, :, 0]  # h
    result_hsv[:, :, 1] = blend_hsv[:, :, 1]  # s
    result_hsv[:, :, 2] = base_hsv[:, :, 2]  # v
    # 转换回 BGR 颜色空间
    result_bgr = cv.cvtColor(result_hsv, cv.COLOR_HSV2BGR)
    # 合并 Alpha 通道
    result = cv.merge([result_bgr, result_a])
    return result


def color(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    颜色

    Note:
        颜色效果
        c.h, c.s, c.v = a.h, a.s, b.v
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 拆分 Alpha 通道
    blend_a = blend_color[:, :, 3]
    base_a = base_color[:, :, 3]
    # 混合 Alpha 通道
    result_a = cv.addWeighted(blend_a, 0.5, base_a, 0.5, 0)
    # 拆分 BGR 通道
    blend_bgr = blend_color[:, :, :3]
    base_bgr = base_color[:, :, :3]
    # 转换到 HSV 颜色空间
    blend_hsv = cv.cvtColor(blend_bgr, cv.COLOR_BGR2HSV)
    base_hsv = cv.cvtColor(base_bgr, cv.COLOR_BGR2HSV)
    # 颜色处理
    result_hsv = np.zeros_like(blend_hsv, dtype=np.uint8)
    result_hsv[:, :, 0] = blend_hsv[:, :, 0]  # h
    result_hsv[:, :, 1] = blend_hsv[:, :, 1]  # s
    result_hsv[:, :, 2] = base_hsv[:, :, 2]  # v
    # 转换回 BGR 颜色空间
    result_bgr = cv.cvtColor(result_hsv, cv.COLOR_HSV2BGR)
    # 合并 Alpha 通道
    result = cv.merge([result_bgr, result_a])
    return result


def luminosity(blend: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    亮度

    Note:
        亮度效果
        c.h, c.s, c.v = b.h, b.s, a.v
        
    Args:
        blend (np.ndarray): 混合图像
        base (np.ndarray): 基础图像

    Returns:
        np.ndarray: 计算后的结果图像
    """
    # 通用颜色空间转换
    blend_color = utils.universalColor(blend)
    base_color = utils.universalColor(base)
    # 拆分 Alpha 通道
    blend_a = blend_color[:, :, 3]
    base_a = base_color[:, :, 3]
    # 混合 Alpha 通道
    result_a = cv.addWeighted(blend_a, 0.5, base_a, 0.5, 0)
    # 拆分 BGR 通道
    blend_bgr = blend_color[:, :, :3]
    base_bgr = base_color[:, :, :3]
    # 转换到 HSV 颜色空间
    blend_hsv = cv.cvtColor(blend_bgr, cv.COLOR_BGR2HSV)
    base_hsv = cv.cvtColor(base_bgr, cv.COLOR_BGR2HSV)
    # 亮度处理
    result_hsv = np.zeros_like(blend_hsv, dtype=np.uint8)
    result_hsv[:, :, 0] = base_hsv[:, :, 0]  # h
    result_hsv[:, :, 1] = base_hsv[:, :, 1]  # s
    result_hsv[:, :, 2] = blend_hsv[:, :, 2]  # v
    # 转换回 BGR 颜色空间
    result_bgr = cv.cvtColor(result_hsv, cv.COLOR_HSV2BGR)
    # 合并 Alpha 通道
    result = cv.merge([result_bgr, result_a])
    return result


if __name__ == "__main__":
    blend = cv.imread(r'./Data/gril01.png', cv.IMREAD_REDUCED_COLOR_2)
    base = cv.imread(r'./Data/gril02.jpg', cv.IMREAD_REDUCED_COLOR_2)
    res = multiply(blend, base)
    cv.imshow("res", res)
    cv.waitKey(0)
    cv.destroyAllWindows()
