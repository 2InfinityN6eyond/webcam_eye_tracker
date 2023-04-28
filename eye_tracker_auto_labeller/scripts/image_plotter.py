import cv2
import numpy as np
import sys 
import random
from PyQt6 import QtWidgets, QtCore, QtGui

class ImagePlotter(QtWidgets.QLabel) :
    
    def __init__(
        self,
        width = 640,
        height = 480
    ) :
        super(ImagePlotter, self).__init__()
        self.width = width
        self.height = height
        self.resize(width, height)

    def update(self, image:np.ndarray) :
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(
            image.data, w, h,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888
        ).scaled(self.width, self.height, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.setPixmap(QtGui.QPixmap.fromImage(q_image))

        