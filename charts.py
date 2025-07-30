"""Charts for Photoshop"""

import sys
from enum import Enum, auto

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, QSizeF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPalette, QPen, QWheelEvent, QMouseEvent, QKeyEvent
from PySide6.QtWidgets import (QApplication, QDockWidget, QGraphicsScene, QGraphicsView, QMainWindow, QPushButton, QRubberBand, QVBoxLayout, QWidget)
import utils


class AddColorMode(Enum):
    """
    颜色混合类型

    - OVERLAP
        - 红色 + 绿色 = (255, 255, 255) - 黄色
        - 红色 + 蓝色 = (255, 255, 255) - 紫色
        - 绿色 + 蓝色 = (255, 255, 255) - 青色
        - 红色 + 绿色 + 蓝色 = 黑色
    - MIX
        - 红色 + 绿色 = 黄色
        - 红色 + 蓝色 = 紫色
        - 绿色 + 蓝色 = 青色
        - 红色 + 绿色 + 蓝色 = 白色
    """
    # 重叠
    OVERLAP = auto()
    # 混合
    MIX = auto()


class HistogramGraphicsView(QGraphicsView):
    """
    图像直方图视图
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # 橡皮筋选择框
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        # 设置样式表，注意，QRubberBand 为非窗口控件，依附于其它窗口，无法使用样式表设置 background-color，只能使用 selection-background-color 设置背景颜色
        self.rubberBand.setStyleSheet('selection-background-color: rgba(0, 0, 0, 128);')
        # 橡皮筋选择框的起点
        self.origin = QPoint()

    # Qt::MouseButton QMouseEvent::button() const: 返回导致事件的按钮。请注意，对于鼠标移动事件，返回的值始终为 Qt::NoButton
    # Qt::MouseButtons QMouseEvent::buttons() const: 返回生成事件时的按钮状态，按钮状态是使用 or 运算符连接 Qt::LeftButton、Qt::RightButton、Qt::MidButton 之一的组合，对于鼠标移动事件，这是按下的所有按钮，对于鼠标按下和双击事件，这包括导致事件的按钮 button，对于鼠标释放事件，这不包括导致事件的按钮 button

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        鼠标按下事件

        Args:
            event (QMouseEvent): 鼠标事件
        """
        # 鼠标左键按下
        if event.button() == Qt.MouseButton.LeftButton:
            # 初始化橡皮筋选择框的起点
            self.origin = event.position().toPoint()
            # 默认 y 轴始终为 0，即窗口顶部
            self.origin.setY(0)
            # 设置橡皮筋选择框的起点和大小
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            # 显示橡皮筋选择框
            self.rubberBand.show()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        鼠标移动事件

        Args:
            event (QMouseEvent): 鼠标事件
        """
        # 鼠标左键按下并移动
        if event.buttons() & Qt.MouseButton.LeftButton:
            # 计算橡皮筋选择框的大小
            rect = QRect(self.origin, QPoint(event.position().x(), self.height())).normalized()
            # 设置橡皮筋选择框的大小
            self.rubberBand.setGeometry(rect)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """
        鼠标释放事件

        Args:
            event (QMouseEvent): 鼠标事件
        """
        # 鼠标左键释放
        if event.button() == Qt.MouseButton.LeftButton:
            # 隐藏橡皮筋选择框
            self.rubberBand.hide()


class HistogramDockWidget(QDockWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # 初始化直方图视图
        self.graphicsView = HistogramGraphicsView(self)
        # 隐藏滚动条
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 初始化直方图场景
        self.scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        # 固定 graphicsView 大小
        self.graphicsView.setFixedSize(256, 128)

        # 添加垂直布局
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.graphicsView)

        # 添加主窗口
        self.widget = QWidget(self)
        self.widget.setLayout(self.layout)
        self.setWidget(self.widget)

        # 固定窗口大小
        self.setFixedSize(256 + 20, 128 + 40)

    def showHistogram(self, image: np.ndarray, modes: AddColorMode = AddColorMode.MIX) -> None:
        """
        显示直方图
        
        灰色图像仅显示单个通道的直方图

        彩色图像显示所有通道（仅为 3 通道，丢弃 Alpha 通道）的直方图
        
        重叠部分将显示为混合颜色
        
        混合颜色模式如下：
        
            - OVERLAP
                - 红色 + 绿色 = (255, 255, 255) - 黄色
                - 红色 + 蓝色 = (255, 255, 255) - 紫色
                - 绿色 + 蓝色 = (255, 255, 255) - 青色
                - 红色 + 绿色 + 蓝色 = 黑色
            - MIX
                - 红色 + 绿色 = 黄色
                - 红色 + 蓝色 = 紫色
                - 绿色 + 蓝色 = 青色
                - 红色 + 绿色 + 蓝色 = 白色

        Args:
            image (np.ndarray): 输入图像
            modes (AddColorMode, optional): 颜色混合类型. Defaults to AddColorMode.MIX.
        """
        # 判断图像类型
        channels = utils.cheakImageType(image)
        # 计算直方图数组
        # 灰度图
        if channels == 0:
            color = [(0, 0, 0)]
            hists = np.array(cv.calcHist([image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])).reshape(1, -1)
        # 彩色图
        elif channels == 3 or channels == 4:
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB) if channels == 3 else cv.cvtColor(image, cv.COLOR_BGRA2RGB)
            hists = np.array([cv.calcHist([image], channels=[i], mask=None, histSize=[256], ranges=[0, 256]) for i in range(3)]).reshape(3, -1)
        # 无效图像
        else:
            return

        # 清空直方图场景
        self.scene.clear()

        if modes == AddColorMode.OVERLAP:

            # 遍历每个直方图
            for h, c in zip(hists, color):
                # 遍历每个直方图的 bin
                for i in range(256):
                    # 获取直方图的最大高度
                    max_height = max([max(h) for h in hists])
                    # 将每个矩形的高度归一化到 [0, 1]，然后乘以 QGraphicsView 的高度
                    height = h[i] / max_height * self.graphicsView.height()
                    # 将每个矩形的宽度设置为 QGraphicsView 的宽度除以直方图的 bin 数量
                    width = self.graphicsView.width() / 256
                    # 添加矩形到直方图场景
                    rect = self.scene.addRect(i * width, self.graphicsView.height() - height, width, height)
                    rect.setBrush(QBrush(QColor(*c)))
                    rect.setOpacity(0.15)

        elif modes == AddColorMode.MIX:

            def mixColor(c1=(0, 0, 0), c2=(0, 0, 0), c3=(0, 0, 0), weights=(1, 1, 1)) -> tuple:
                """
                混合三个 RGB 颜色，返回混合后的颜色
                ---
                
                :param c1: 第一个颜色，格式为 (R, G, B)
                :param c2: 第二个颜色，格式为 (R, G, B)
                :param c3: 第三个颜色，格式为 (R, G, B)
                :param weights: 每个颜色的权重，格式为 (w1, w2, w3)
                :return: 混合后的颜色，格式为 (R, G, B)
                """
                # 将颜色和权重转换为 numpy 数组
                colors = np.array([c1, c2, c3])
                weights = np.array(weights)
                # 计算加权平均
                mixed_color = np.average(colors, axis=0, weights=weights)
                # 确保颜色值在 [0, 255] 范围内
                mixed_color = np.clip(mixed_color, 0, 255).astype(int)
                return tuple(mixed_color)

            # 遍历每个直方图的 bin
            for i in range(256):
                # 获取每个直方图当前 bin 的大小顺序，并将其转换为 numpy 数组，以便接下来的掩码操作
                sizes = np.array([h[i] for h in hists])
                order = np.argsort(sizes)

                # 遍历每个直方图
                for h, c in zip(hists, color):
                    # 获取直方图的最大高度
                    max_height = max([max(h) for h in hists])

                    # 灰度图
                    if hists.shape[0] == 1:
                        # 获取每个颜色的高度位置
                        height = sizes
                        # 将每个矩形的高度归一化到 [0, 1]，然后乘以 QGraphicsView 的高度
                        height = height / max_height * self.graphicsView.height()
                        # 将每个矩形的宽度设置为 QGraphicsView 的宽度除以直方图的 bin 数量
                        width = self.graphicsView.width() / 256
                        # 添加矩形到直方图场景
                        rect = self.scene.addRect(i * width, self.graphicsView.height() - height, width, height)
                        rect.setBrush(QBrush(QColor(*c)))
                        rect.setOpacity(0.15)

                    # 颜色图
                    elif hists.shape[0] == 3 or hists.shape[0] == 4:
                        # 获取每个颜色的高度位置
                        lower_height, middle_height, upper_height = sizes[order]
                        # 将每个矩形的高度归一化到 [0, 1]，然后乘以 QGraphicsView 的高度
                        lower_height = lower_height / max_height * self.graphicsView.height()
                        middle_height = middle_height / max_height * self.graphicsView.height()
                        upper_height = upper_height / max_height * self.graphicsView.height()
                        # 将每个矩形的宽度设置为 QGraphicsView 的宽度除以直方图的 bin 数量
                        width = self.graphicsView.width() / 256
                        # 依据高度位置混合颜色，并添加矩形到直方图场景
                        lower_color, middle_color, upper_color = np.array(color)[order]
                        # 数组级别比较，判断颜色是否相等
                        if np.array_equal(c, lower_color):
                            lower_color = mixColor(lower_color, middle_color, upper_color)
                            rect = self.scene.addRect(i * width, self.graphicsView.height() - lower_height, width, lower_height)
                            rect.setBrush(QBrush(QColor(*lower_color)))
                            rect.setOpacity(0.15)
                        # 数组级别比较，判断颜色是否相等
                        elif np.array_equal(c, middle_color):
                            middle_color = mixColor(middle_color, upper_color)
                            rect = self.scene.addRect(i * width, self.graphicsView.height() - middle_height, width, middle_height - lower_height)
                            rect.setBrush(QBrush(QColor(*middle_color)))
                            rect.setOpacity(0.15)
                        # 数组级别比较，判断颜色是否相等
                        elif np.array_equal(c, upper_color):
                            upper_color = mixColor(upper_color)
                            rect = self.scene.addRect(i * width, self.graphicsView.height() - upper_height, width, upper_height - middle_height)
                            rect.setBrush(QBrush(QColor(*upper_color)))
                            rect.setOpacity(0.15)


if __name__ == "__main__":
    app = QApplication([])

    class ApplicationWindow(QMainWindow):

        def __init__(self):
            super().__init__()
            self.histogramDockWidget = HistogramDockWidget(self)
            self.addDockWidget(Qt.RightDockWidgetArea, self.histogramDockWidget)
            image = cv.imread(r'./Data/girl.jpg')
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.histogramDockWidget.showHistogram(image, modes=AddColorMode.MIX)
            # self.histogramDockWidget.showHistogram(gray)

    mainWindow = ApplicationWindow()
    mainWindow.show()

    sys.exit(app.exec())
