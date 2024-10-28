'''MainWindow for Photoshop'''

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel, QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPixmapItem
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDragLeaveEvent, QDropEvent, QImage, QPixmap, QPicture
from photoshop_ui import Ui_PhotoShop
import numpy as np
import utils
import imgproc
import charts


class Photoshop(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_PhotoShop()
        self.ui.setupUi(self)

        # 启动拖拽
        self.setAcceptDrops(True)
        self.ui.graphicsView.setAcceptDrops(True)

        # 初始化场景
        self.graphicScene = QGraphicsScene()
        self.ui.graphicsView.setScene(self.graphicScene)

        # 文件读取器
        self.reader = utils.Reader()

        # 文件路径
        self.file: str = ''

        # 当前图像，暂存，用以加速图像处理 : cv.Mat
        self.image: np.ndarray = None
        # 当前图形项
        self.pixmapItem: QGraphicsPixmapItem = None
        # TODO: 当前视频
        self.qVideos = None
        # TODO: 当前音频
        self.qAudios = None

        # ------------------------------------ 窗口组件 ------------------------------------
        # 图像直方图
        self.histogramDockWidget = charts.HistogramDockWidget(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.histogramDockWidget)

        # ------------------------------------ 连接信号 ------------------------------------
        # 打开文件
        self.ui.actionOpen.triggered.connect(lambda: self.openFile())
        # 灰度
        self.ui.actionGray.triggered.connect(lambda: self.gray(self.image))
        # 黑白
        self.ui.actionMonochrome.triggered.connect(lambda: self.monochrome(self.image))
        # 反相
        self.ui.actionInversion.triggered.connect(lambda: self.inversion(self.image))
        # 阈值
        self.ui.actionThresholds.triggered.connect(lambda: self.thresholds(self.image))
        # 色调均化
        self.ui.actionToneHomogenization.triggered.connect(lambda: self.toneHomogenization(self.image))

    # ------------------------------------ 重载 ------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        '''拖拽进入事件'''
        # 若拖拽数据中包含 URL，则接受拖拽
        if event.mimeData().hasUrls():
            # 接受建议动作：默认为 CopyAction
            event.acceptProposedAction()
        # 否则忽略拖拽
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        '''拖拽移动事件'''
        # 若拖拽数据中包含 URL，则接受拖拽
        if event.mimeData().hasUrls():
            # 接受建议动作：默认为 CopyAction
            event.acceptProposedAction()
        # 否则忽略拖拽
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        '''拖拽离开事件'''

    def dropEvent(self, event: QDropEvent) -> None:
        '''拖拽释放事件'''
        # 遍历拖拽数据中的 URL
        for url in event.mimeData().urls():
            # 若 URL 为本地文件
            if url.isLocalFile():
                # 获取文件路径
                self.file = url.toLocalFile()
                # 发送加载文件信号
                self.openFile(self.file)

    # ------------------------------------ 成员函数 ------------------------------------
    def updateScene(self) -> None:
        '''更新场景'''
        # 如果读取器状态改变，清空场景
        if self.reader.stateChanged:
            self.graphicScene.clear()
            self.reader.stateChanged = False
        # QMessageBox.information(self, '提示', f'{self.reader.state}')

    def showImage(self, image: QImage) -> None:
        '''显示图片'''
        # 若图片为 None，则返回
        if image is None:
            return

        # 更新场景
        self.updateScene()

        # 位图对象
        pixmap = QPixmap.fromImage(image)
        # cv.Mat 对象
        self.image = utils.QImage2Mat(image)

        # 显示图片
        if self.pixmapItem is None:  # 第一次显示图片
            self.pixmapItem = self.graphicScene.addPixmap(pixmap)
        else:  # 更新图片
            self.pixmapItem.setPixmap(pixmap)

        # 显示直方图
        self.histogramDockWidget.showHistogram(self.image)

    def showVideo(self, video):
        '''显示视频'''
        # 如果视频为 None，则返回
        if video is None:
            return

        # 更新场景
        self.updateScene()

        # TODO: 播放视频

    def showAudio(self, audio):
        '''显示音频'''
        # 如果音频为 None，则返回
        if audio is None:
            return

        # 更新场景
        self.updateScene()

        # TODO: 播放音频

    # ------------------------------------ 槽函数 ------------------------------------
    @QtCore.Slot(str, name='openFile', result=None)
    def openFile(self, file: str = '') -> None:
        '''打开文件，默认文件为空，表示打开文件对话框选择文件'''
        # 读取文件路径
        self.file = file
        # 若文件路径为空，则打开文件对话框选择文件
        if self.file == '':
            self.file, self.nameFilter = QFileDialog.getOpenFileName(
                self, '打开文件', '',
                '所有格式(*.*);;JPEG(*.JPG *.JPEG *JPE);;JPEG 2000(*.JPF *JPX *JP2 *J2C *J2K *JPC);;JPEG 立体(*.JPS);;PNG(*.PNG);;SVG(*.SVG *.SVGZ);;TIFF(*.TIF *.TIFF);;WebP(*.WEBP);;视频(*.264 *.3GP *.3GPP *.AVC *.AVI *.F4V *.FLV *.M4V *.MOV *.MP4 *.MPE *.MPEG *.MPG *.MTS *.MXF *.R3D *.TS *.VOB *.WM *.WMV);;音频(*.AAC *.AC3 *.M2A *.M4A *.MP2 *.MP3 *.WMA *.WM)'
            )
            # 若未选择文件，则弹出警告对话框
            if self.file == '':
                QMessageBox.warning(self, '警告', '未选择文件')
                return

        # 根据文件类型加载文件
        item = self.reader.readFile(self.file)

        # 若未加载成功，则返回
        if item is None:
            return

        # 根据读取状态显示文件
        if self.reader.state == utils.ReadState.IMAGE:
            self.showImage(item)
        elif self.reader.state == utils.ReadState.VIDEO:
            self.showVideo(item)
        elif self.reader.state == utils.ReadState.AUDIO:
            self.showAudio(item)

    @QtCore.Slot(np.ndarray, name='gray', result=None)
    def gray(self, image: np.ndarray) -> None:
        '''灰度'''
        # 灰度处理
        grayImage = imgproc.gray(image)
        # 如果灰度图像为空，则返回
        if grayImage is None:
            return
        else:
            self.showImage(utils.Mat2QImage(grayImage))

    @QtCore.Slot(np.ndarray, name='monochrome', result=None)
    def monochrome(self, image: np.ndarray) -> None:
        '''黑白'''
        # 黑白处理
        monochromeImage = imgproc.monochrome(image)
        # 如果黑白图像为空，则返回
        if monochromeImage is None:
            return
        else:
            self.showImage(utils.Mat2QImage(monochromeImage))

    @QtCore.Slot(np.ndarray, name='inversion', result=None)
    def inversion(self, image: np.ndarray) -> None:
        '''反相'''
        # 反相处理
        inversionImage = imgproc.inversion(image)
        # 如果反相图像为空，则返回
        if inversionImage is None:
            return
        else:
            self.showImage(utils.Mat2QImage(inversionImage))

    @QtCore.Slot(np.ndarray, name='thresholds', result=None)
    def thresholds(self, image: np.ndarray) -> None:
        '''阈值'''
        # 阈值处理
        thresholdsImage = imgproc.thresholds(image)
        # 如果阈值图像为空，则返回
        if thresholdsImage is None:
            return
        else:
            self.showImage(utils.Mat2QImage(thresholdsImage))

    @QtCore.Slot(np.ndarray, name='toneHomogenization', result=None)
    def toneHomogenization(self, image: np.ndarray) -> None:
        '''色调均化'''
        # 色调均化处理
        toneHomogenizationImage = imgproc.toneHomogenization(image)
        # 如果色调均化图像为空，则返回
        if toneHomogenizationImage is None:
            return
        else:
            self.showImage(utils.Mat2QImage(toneHomogenizationImage))
