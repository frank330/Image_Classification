# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QMainWindow
import sys
from PIL import Image
from tensorflow.python.keras.models import load_model
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QDesktopWidget, QGraphicsPixmapItem, QFileDialog, QGraphicsScene, QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from graphics import GraphicsView, GraphicsPixmapItem
import cv2
import numpy as np
from Image import Image
global imgName


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # 设置主屏幕
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(825, 570)
        # 获取屏幕的尺寸信息，也可以理解为屏幕的分辨率信息。获取到的屏幕信息有两个属性，一个是width对应屏幕的长度，一个是height对应屏幕的宽度
        center = QDesktopWidget().screenGeometry()
        # 将主窗口置于屏幕中间
        MainWindow.move((center.width() - 825) / 2, (center.height() - 570) / 2)
        # 定义右边三个按钮的垂直布局
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        # setGeometry（）设置窗口的位置
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(620, 190, 151, 341))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget_2")
        # QVBoxLayout 垂直布局管理
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        # setContentsMargins外边距左 上 右 下
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_1 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_1.setObjectName("pushButton_1")
        self.verticalLayout_2.addWidget(self.pushButton_1)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_2.addWidget(self.pushButton_3)

        # 定义图片显示窗口
        self.graphicsView = GraphicsView(self.centralwidget)
        self.graphicsView.setEnabled(True)
        self.graphicsView.setGeometry(QtCore.QRect(30, 180, 552, 352))
        self.graphicsView.setObjectName("graphicsView")

        # 定义上方的结果显示框
        # QTextEdit类提供了一个用于编辑和显示纯文本和富文本的小部件。
        self.label = QtWidgets.QTextEdit(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(31, 100, 550, 52))
        self.label.setObjectName("label")
        self.label.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff) #设置垂直滚动条
        MainWindow.setCentralWidget(self.centralwidget)
        # 以上中心界面创建完毕



        # 信号槽
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        MainWindow.setStyleSheet("QMainWindow{background:#E9F2FF}")

        self.graphicsView.setStyleSheet("QGraphicsView{background:#E9F2FF}")
        # self.graphicsView.setLineWidth(3) # 设置外线宽度
        # self.graphicsView.setMidLineWidth(5) # 设置中线宽度

        self.pushButton_1.setText(_translate("MainWindow", "打开图片"))
        self.pushButton_1.setIcon(QtGui.QIcon("icons/open.svg"))
        self.pushButton_1.setIconSize(QtCore.QSize(40, 40))
        self.pushButton_1.setStyleSheet("QPushButton{background:#16A085;border:none;color:#000000;font-size:15px;}"
                                        "QPushButton:hover{background-color:#008080;}")
        self.pushButton_1.clicked.connect(self.clickOpen)

        self.pushButton_2.setText(_translate("MainWindow", "图像识别"))
        self.pushButton_2.setIcon(QtGui.QIcon("icons/c.svg"))
        self.pushButton_2.setIconSize(QtCore.QSize(40, 40))
        self.pushButton_2.setStyleSheet("QPushButton{background:#9F35FF;border:none;color:#000000;font-size:15px}"
                                        "QPushButton:hover{background-color:#9932CC;}")
        self.pushButton_2.clicked.connect(self.recognition)


        self.pushButton_3.setText(_translate("MainWindow", "退出系统"))
        self.pushButton_3.setIcon(QtGui.QIcon("icons/close_.svg"))
        self.pushButton_3.setIconSize(QtCore.QSize(40, 40))
        self.pushButton_3.setStyleSheet("QPushButton{background:#CE0000;border:none;color:#000000;font-size:15px;}"
                                      "QPushButton:hover{background-color:#8B0000;}")
        self.pushButton_3.clicked.connect(self.close)


        self.label.setText("识别结果")
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setMidLineWidth(5)
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(13)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)


        self.messageBox = QMessageBox()
        self.messageBox.setStyleSheet("QMessageBox{background-color:#CE0000;border:none;color:#000000;font-size:15px;}")

    def close(self):
        reply = self.messageBox.question(None, "Quit", "确定要关闭该程序？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit()

    def clickOpen(self):
        global imgName
        imgName, imgType = QFileDialog.getOpenFileName(None, "打开图片", "", "*.jpg;;*.png;;*.jpeg;;All Files(*)")
        self.ifile = imgName
        img = cv2.imread(imgName)
        img1 = cv2.resize(img, (350, 550))
        self.image = Image(img1)
        self.img = self.image.pos_img
        H, W, C = self.image.img.shape
        P = 3 * W
        qimage = QImage(self.image.img.data, W, H, P, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        self.graphicsView.setSceneRect(0, 0, 550,350)
        self.graphicsView.setItem(pixmap)
        self.graphicsView.Scene()
        self.graphicsView.setStyleSheet("QGraphicsView{background-color:#66B3FF}")


    def recognition(self):
        global imgName


        model_ = load_model(r"./model/model.h5")
        img = cv2.imread(imgName)
        img = img.reshape((1, 64,64, 3))
        img = img.astype('float32')
        img = img / 255.0
        result = model_.predict(img)  # 测算一下该img属于某个label的概率
        max_index = np.argmax(result)  # 找出概率最高的
        label = ["这是狗。","这是马。","这是大象。","这是蝴蝶。","这是母鸡。","这是猫。","这是牛。","这是羊。","这是蜘蛛。","这是松鼠。"]
        self.label.setText(label[max_index])
        self.label.setAlignment(Qt.AlignCenter)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setWindowTitle("图像识别系统")
    MainWindow.show()
    sys.exit(app.exec_())
