"""Utils Tools"""

from enum import Enum, auto
from typing import Any

import cv2 as cv
import numpy as np
from PIL import Image
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QMessageBox, QWidget


def cheakImageType(image: np.ndarray) -> int:
    """
    检测图像类型，若为灰度图则返回 0，若为彩色图则返回 3 或 4

    Args:
        image (np.ndarray): 输入图像

    Returns:
        int: 通道数
    """
    try:
        channels = image.shape[2]
    except IndexError:
        channels = 0
    return channels


def universalColor(image: np.ndarray) -> np.ndarray | None:
    """
    通用颜色空间转换，将任意图像转换为通用的 BGRA 格式

    Args:
        image (np.ndarray): 输入图像

    Returns:
        np.ndarray | None: 通用颜色空间图像
    """
    # 判断图像类型
    channels = cheakImageType(image)
    # 灰度图
    if channels == 0:
        return cv.cvtColor(image, cv.COLOR_GRAY2BGRA)
    # 彩色图
    elif channels == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2BGRA)
    # 带有 Alpha 通道的彩色图
    elif channels == 4:
        return image.copy()


def Pillow2Mat(image: Image.Image) -> np.ndarray | None:
    """
    PIL 转换为 Mat

    Args:
        image (Image.Image): 输入图像

    Returns:
        np.ndarray | None: 输出图像
    """
    return cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)


def QImage2Mat(image: QImage) -> np.ndarray | None:
    """
    QImage 转换为 Mat

    Args:
        image (QImage): 输入图像

    Returns:
        np.ndarray | None: 输出图像
    """
    switcher = {
        # 8 位索引存储图像
        QImage.Format.Format_Indexed8:
        lambda: np.array(image.constBits()).reshape(image.height(), image.width()).astype(np.uint8),
        # 8 位灰度图存储图像
        QImage.Format.Format_Grayscale8:
        lambda: np.array(image.constBits()).reshape(image.height(), image.width()).astype(np.uint8),
        # 24 位 RGB 格式(8-8-8)存储图像，这是一种字节顺序格式，强制读取顺序为 R、G、B，这意味着 24 位编码在大端和小端架构之间有所不同，分别是(0xRRGGBB)和(0xBBGGRR)。如果读取为字节 0xRR、0xGG、0xBB，则任何体系结构上的颜色顺序都是相同的
        QImage.Format.Format_RGB888:
        lambda: cv.cvtColor(np.array(image.constBits()).reshape(image.height(), image.width(), 3).astype(np.uint8), cv.COLOR_RGB2BGR),
        # 32 位 RGB 格式(0xFFRRGGBB)存储图像，无论在大端还是小端架构上，颜色顺序都是相同的(0xFFRRGGBB)
        QImage.Format.Format_RGB32:
        lambda: np.array(image.constBits()).reshape(image.height(), image.width(), 4).astype(np.uint8),
        # 32 位 ARGB 格式(0xAARRGGBB)存储图像，无论在大端还是小端架构上，颜色顺序都是相同的(0xAARRGGBB)
        QImage.Format.Format_ARGB32:
        lambda: np.array(image.constBits()).reshape(image.height(), image.width(), 4).astype(np.uint8),
        #  预乘的 32 位 ARGB 格式(0xAARRGGBB)存储图像，无论在大端还是小端架构上，颜色顺序都是相同的(0xAARRGGBB)。图像使用预乘的 32 位 ARGB 格式(0xAARRGGBB)存储，即红色、绿色和蓝色通道乘以 alpha 分量除以 255(如果 RR、GG 或 BB 的值高于 Alpha 通道，则结果未定义)。某些操作(例如使用 alpha 混合的图像合成)使用预乘 ARGB32 比使用普通 ARGB32 更快
        QImage.Format.Format_ARGB32_Premultiplied:
        lambda: np.array(image.constBits()).reshape(image.height(), image.width(), 4).astype(np.uint8),
        # 32 位字节排序的 RGBA 格式(0xRRGGBBAA)存储图像，与 QImage::Format_ARGB32 不同，这是一种字节顺序格式，强制读取顺序为 R、G、B、A，这意味着 32 位编码在大端和小端架构之间有所不同，分别是(0xRRGGBBAA)和(0xAABBGGRR)。如果读取为字节 0xRR、0xGG、0xBB、0xAA，则任何体系结构上的颜色顺序都是相同的
        QImage.Format.Format_RGBA8888:
        lambda: cv.cvtColor(np.array(image.constBits()).reshape(image.height(), image.width(), 4).astype(np.uint8), cv.COLOR_RGBA2BGRA),
        # 预乘的 32 位字节排序的 RGBA 格式(0xRRGGBBAA)存储图像，与 QImage::Format_ARGB32_Premultiplied 不同，这是一种字节顺序格式，强制读取顺序为 R、G、B、A，这意味着 32 位编码在大端和小端架构之间有所不同，分别是(0xRRGGBBAA)和(0xAABBGGRR)。如果读取为字节 0xRR、0xGG、0xBB、0xAA，则任何体系结构上的颜色顺序都是相同的。图像使用预乘的 32 位字节顺序 RGBA 格式(8-8-8-8)存储，即红色、绿色和蓝色通道乘以 alpha 分量除以 255(如果 RR、GG 或 BB 的值高于 Alpha 通道，则结果未定义)。某些操作(例如使用 alpha 混合的图像合成)使用预乘 RGBA8888 比使用普通 RGBA8888 更快
        QImage.Format.Format_RGBA8888_Premultiplied:
        lambda: cv.cvtColor(np.array(image.constBits()).reshape(image.height(), image.width(), 4).astype(np.uint8), cv.COLOR_RGBA2BGRA),
    }
    return switcher.get(image.format(), lambda: None)()


def Mat2QImage(mat: np.ndarray) -> QImage | None:
    """
    Mat 转换为 QImage

    Args:
        mat (np.ndarray): 输入图像

    Returns:
        QImage | None: 输出图像
    """
    switcher = {
        # 8 位无符号整数，channel = 1，对应 QImage::Format_Indexed8 或 QImage::Format_Grayscale8，与 C++ 版本不同，Pyside6 中，QImage::Format_Indexed8 不需要手动设置调色板，Pyside6 简化了操作，使 QImage 自动创建灰度调色板
        0: lambda: QImage(mat.data, mat.shape[1], mat.shape[0], QImage.Format.Format_Indexed8),
        # 8 位无符号整数，channel = 3，对应 QImage::Format_RGB888
        3: lambda: QImage(mat.data, mat.shape[1], mat.shape[0], QImage.Format.Format_RGB888).rgbSwapped(),
        # 8 位无符号整数，channel = 4，对应 QImage::Format_ARGB32 或 QImage::Format_ARGB32_Premultiplied，或交换 R、B 通道后的 QImage::Format_RGBA8888 或 QImage::Format_RGBA8888_Premultiplied
        4: lambda: QImage(mat.data, mat.shape[1], mat.shape[0], QImage.Format.Format_ARGB32),
    }
    channels = cheakImageType(mat)
    return switcher.get(channels, lambda: None)()


class ReadState(Enum):
    """
    读取状态枚举
    """
    NOSTATE = auto()
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()


class Reader:
    """
    文件读取器
    """

    def __init__(self):
        # 全部文件读取器
        self.readers = {
            # https://docs.opencv.org/5.x/d4/da8/group__imgcodecs.html
            'cv_image':
            ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'webp', 'avif', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm', 'pfm', 'sr', 'ras', 'tiff', 'tif', 'exr', 'hdr', 'pic'],
            'pil_image': [
                'blp', 'bmp', 'bufr', 'cur', 'dcx', 'dds', 'dib', 'eps', 'ps', 'fit', 'fits', 'flc', 'fli', 'ftc', 'ftu', 'gbr', 'gif', 'grib', 'h5', 'hdf', 'icns', 'ico', 'im',
                'iim', 'jfif', 'jpe', 'jpeg', 'jpg', 'j2c', 'j2k', 'jp2', 'jpc', 'jpf', 'jpx', 'mpeg', 'mpg', 'msp', 'pcd', 'pcx', 'pxr', 'apng', 'png', 'pbm', 'pfm', 'pgm', 'pnm',
                'ppm', 'psd', 'qoi', 'bw', 'rgb', 'rgba', 'sgi', 'ras', 'icb', 'tga', 'vda', 'vst', 'tif', 'tiff', 'webp', 'emf', 'wmf', 'xbm', 'xpm'
            ],
            #
            'qt_image': [],
            #
            'qt_video': [],
            #
            'qt_audio': [],
        }
        # 当前文件读取器
        self.reader = None

        # 全部读取状态
        self.states: list[ReadState] = [ReadState.NOSTATE, ReadState.IMAGE, ReadState.VIDEO, ReadState.AUDIO]
        # 当前读取状态
        self.state: ReadState = ReadState.NOSTATE
        # 状态改变
        self.stateChanged: bool = False

    def getFileFormat(self, file: str) -> str:
        """
        获取文件格式

        Args:
            file (str): 文件路径

        Returns:
            str: 文件格式
        """
        return file.split('.')[-1].lower()

    def getFileName(self, file: str) -> str:
        """
        获取文件名称

        Args:
            file (str): 文件路径

        Returns:
            str: 文件名称
        """
        return file.split('/')[-1]

    def isSupported(self, file: str) -> bool:
        """
        检查文件是否支持，若支持则返回 True，并设置文件读取器，若不支持则返回 False

        Args:
            file (str): 文件路径

        Returns:
            bool: 文件是否支持
        """
        for key in self.readers:
            fileFormat = self.getFileFormat(file)
            if fileFormat in self.readers[key]:
                self.reader = key
                return True
        self.reader = None
        return False

    def setState(self, state: ReadState) -> None:
        """
        设置读取状态

        Args:
            state (ReadState): 读取状态

        Raises:
            ValueError: Unsupported state
        """
        # 若状态不在全部状态中，则抛出异常
        if state not in self.states:
            raise ValueError('Unsupported state')

        if state == self.state:  # 状态未改变
            self.stateChanged = False
        else:  # 状态改变
            self.state = state
            self.stateChanged = True

    def readFile(self, file: str) -> Any | None:
        """
        读取文件

        Args:
            file (str): 文件路径

        Returns:
            Any | None: 读取结果
        """
        if not self.isSupported(file):  # 文件不支持
            QMessageBox.warning(QWidget(), '警告', '不支持的文件格式')
            return None
        else:  # 文件支持
            if self.reader == 'cv_image':  # cv image reader
                # 设置读取状态
                self.setState(ReadState.IMAGE)
                # 读取图像
                return Mat2QImage(cv.imread(file))
            elif self.reader == 'pil_image':  # pillow image reader
                # 设置读取状态
                self.setState(ReadState.IMAGE)
                # 读取图像
                return Mat2QImage(Pillow2Mat(Image.open(file)))
            elif self.reader == 'qt_image':  # qt image reader
                # 设置读取状态
                self.setState(ReadState.IMAGE)
                # TODO: 读取图像
                return None
            elif self.reader == 'qt_video':  # qt video reader
                # 设置读取状态
                self.setState(ReadState.VIDEO)
                # TODO: 读取视频
                return None
            elif self.reader == 'qt_audio':  # qt audio reader
                # 设置读取状态
                self.setState(ReadState.AUDIO)
                # TODO: 读取音频
                return None
