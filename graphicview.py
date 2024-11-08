'''Graphics View'''

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDragLeaveEvent, QDropEvent, QImage, QPixmap, QPicture, \
    QWheelEvent, QMouseEvent, QKeyEvent


class GraphicView(QGraphicsView):

    def __init__(self, parent=None):
        super().__init__(parent)

    def wheelEvent(self, event: QWheelEvent):
        '''
        鼠标滚轮事件
        '''
        # 缩放因子
        scaleFactor = 1.2
        # 旋转因子
        rotateFactor = 10
        # 移动因子
        translateFactor = 10
        # Ctrl + 滚轮滑动
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # 滚轮向上，放大
            if event.angleDelta().y() > 0:
                self.scale(scaleFactor, scaleFactor)
            # 滚轮向下，缩小
            elif event.angleDelta().y() < 0:
                self.scale(1 / scaleFactor, 1 / scaleFactor)
        # Shift + 滚轮滑动
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # 滚轮向上，左移
            if event.angleDelta().y() > 0:
                # 由于 QGraphicsView 的 alignment 属性默认为 AlignCenter: 0x84，所以在调用 translate() 方法时，QGraphicsView 又会被自动调整到中心对齐，导致 translate() 方法无效，在视觉上为 QGraphicsView 的微小抖动效果
                # self.translate(-translateFactor, 0)
                # print(self.transform())
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - translateFactor)
            # 滚轮向下，右移
            elif event.angleDelta().y() < 0:
                # 由于 QGraphicsView 的 alignment 属性默认为 AlignCenter: 0x84，所以在调用 translate() 方法时，QGraphicsView 又会被自动调整到中心对齐，导致 translate() 方法无效，在视觉上为 QGraphicsView 的微小抖动效果
                # self.translate(translateFactor, 0)
                # print(self.transform())
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + translateFactor)
        # Alt + 滚轮滑动
        elif event.modifiers() & Qt.KeyboardModifier.AltModifier:
            # 滚轮向上，逆时针旋转
            if event.angleDelta().y() > 0:
                self.rotate(-rotateFactor)
            # 滚轮向下，顺时针旋转
            elif event.angleDelta().y() < 0:
                self.rotate(rotateFactor)
        # 滚轮滑动
        else:
            # 滚轮向上，上移
            if event.angleDelta().y() > 0:
                # 由于 QGraphicsView 的 alignment 属性默认为 AlignCenter: 0x84，所以在调用 translate() 方法时，QGraphicsView 又会被自动调整到中心对齐，导致 translate() 方法无效，在视觉上为 QGraphicsView 的微小抖动效果
                # self.translate(0, -translateFactor)
                # print(self.transform())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - translateFactor)
            # 滚轮向下，下移
            elif event.angleDelta().y() < 0:
                # 由于 QGraphicsView 的 alignment 属性默认为 AlignCenter: 0x84，所以在调用 translate() 方法时，QGraphicsView 又会被自动调整到中心对齐，导致 translate() 方法无效，在视觉上为 QGraphicsView 的微小抖动效果
                # self.translate(0, translateFactor)
                # print(self.transform())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() + translateFactor)
        # super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        '''
        键盘事件
        '''
        # 缩放因子
        scaleFactor = 1.2
        # 旋转因子
        rotateFactor = 10
        # 移动因子
        translateFactor = 10
        # + 放大
        if event.key() == Qt.Key.Key_Plus:
            self.scale(scaleFactor, scaleFactor)
        # - 缩小
        elif event.key() == Qt.Key.Key_Minus:
            self.scale(1 / scaleFactor, 1 / scaleFactor)
        # Q 逆时针旋转
        elif event.key() == Qt.Key.Key_Q:
            self.rotate(-rotateFactor)
        # E 顺时针旋转
        elif event.key() == Qt.Key.Key_E:
            self.rotate(rotateFactor)
        # W 上移
        elif event.key() == Qt.Key.Key_W or event.key() == Qt.Key.Key_Up:
            # 由于 QGraphicsView 的 alignment 属性默认为 AlignCenter: 0x84，所以在调用 translate() 方法时，QGraphicsView 又会被自动调整到中心对齐，导致 translate() 方法无效，在视觉上为 QGraphicsView 的微小抖动效果
            # self.translate(0, -translateFactor)
            # print(self.transform())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - translateFactor)
        # S 下移
        elif event.key() == Qt.Key.Key_S or event.key() == Qt.Key.Key_Down:
            # 由于 QGraphicsView 的 alignment 属性默认为 AlignCenter: 0x84，所以在调用 translate() 方法时，QGraphicsView 又会被自动调整到中心对齐，导致 translate() 方法无效，在视觉上为 QGraphicsView 的微小抖动效果
            # self.translate(0, translateFactor)
            # print(self.transform())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + translateFactor)
        # A 左移
        elif event.key() == Qt.Key.Key_A or event.key() == Qt.Key.Key_Left:
            # 由于 QGraphicsView 的 alignment 属性默认为 AlignCenter: 0x84，所以在调用 translate() 方法时，QGraphicsView 又会被自动调整到中心对齐，导致 translate() 方法无效，在视觉上为 QGraphicsView 的微小抖动效果
            # self.translate(-translateFactor, 0)
            # print(self.transform())
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - translateFactor)
        # D 右移
        elif event.key() == Qt.Key.Key_D or event.key() == Qt.Key.Key_Right:
            # 由于 QGraphicsView 的 alignment 属性默认为 AlignCenter: 0x84，所以在调用 translate() 方法时，QGraphicsView 又会被自动调整到中心对齐，导致 translate() 方法无效，在视觉上为 QGraphicsView 的微小抖动效果
            # self.translate(translateFactor, 0)
            # print(self.transform())
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + translateFactor)
        # super().keyPressEvent(event)
