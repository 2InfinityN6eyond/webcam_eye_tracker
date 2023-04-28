import os
import time
import cv2
import sys 
import numpy as np
import multiprocessing
import pickle
from multiprocessing import shared_memory

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

from image_plotter import ImagePlotter
from mediapipe_visualizer import ThreeDimensionVisualizer, TwoDimensionVisualizer
from utils import N_FACE_LANDMARK_FEATURES

class Configs :
    def __init__(
        self,
        proj_root_dir,
        screen_width, screen_height,
        frame_width, frame_height,
        img_shm_name,
        face_shm_name,
        shm_queue_size
    ) :
        self.PROJECT_ROOT_PATH = proj_root_dir
        self.DATA_ROOT_PATH  = os.path.join(self.PROJECT_ROOT_PATH, "data")
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_geometry = np.array([screen_width, screen_height])

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.img_shm_name = img_shm_name
        self.face_shm_name = face_shm_name
        self.shm_queue_size = shm_queue_size

        self.mouse_pos_raw = [None, None]
        self.mouse_pos = [None, None]
        self.scroll_val = [None, None]
        self.click_val = [None, None]

        self.DATA_DIR_PATH = None

class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        proj_root_dir   :str,
        screen_width    :int,
        screen_height   :int,
        frame_width     :int,
        frame_height    :int,
        img_shm_name    :str,
        face_shm_name   :str,
        shm_queue_size  :int,
        save_data_queue :multiprocessing.Queue
    ) :
        super(MainWindow, self).__init__()
        
        self.configs = Configs(
            proj_root_dir,
            screen_width, screen_height,
            frame_width, frame_height,
            img_shm_name,
            face_shm_name,
            shm_queue_size
        )
        self.save_data_queue = save_data_queue
        self.image_shm = shared_memory.SharedMemory(name = img_shm_name)
        self.image_queue = np.ndarray(
            (shm_queue_size, frame_height, frame_width, 3),
            dtype=np.uint8, buffer = self.image_shm.buf
        )

        self.face_shm = shared_memory.SharedMemory(name = face_shm_name)
        self.face_queue = np.ndarray(
            (shm_queue_size, N_FACE_LANDMARK_FEATURES, 3),
            dtype=np.float64, buffer = self.face_shm.buf
        )
        
        self.setMouseTracking(True)
        self.initUI()

    def initUI(self) :
        self.two_dimension_visualizer = TwoDimensionVisualizer()

        self.three_dimention_visualizer = ThreeDimensionVisualizer()
        self.three_dimention_visualizer.setMinimumSize(500, 500)
        self.record_curr_frame_button = QtWidgets.QPushButton(
            "record_curr_frame", self
        )

        self.webcam_image_plotter = ImagePlotter(500, 500)
        self.face_plotter = ImagePlotter(250, 250)
        self.face_vis_plotter = ImagePlotter(250, 250)
        self.left_eye_plotter = ImagePlotter(250, 200)
        self.left_eye_vis_plotter = ImagePlotter(250, 200)
        self.right_eye_plotter = ImagePlotter(250, 200)
        self.right_eye_vis_plotter = ImagePlotter(250, 200)
        
        self.webcam_image_plotter.update(
            np.zeros((500, 500, 3), dtype=np.uint8) + 150
        )
        self.face_plotter.update(
            np.zeros((250, 250, 3), dtype=np.uint8) + 150
        )
        self.face_vis_plotter.update(
            np.zeros((250, 250, 3), dtype=np.uint8) + 150
        )
        self.left_eye_plotter.update(
            np.zeros((250, 100, 3), dtype=np.uint8) + 150
        )
        self.left_eye_vis_plotter.update(
            np.zeros((250, 100, 3), dtype=np.uint8) + 150
        )
        self.right_eye_plotter.update(
            np.zeros((250, 100, 3), dtype=np.uint8) + 150
        )
        self.right_eye_vis_plotter.update(
            np.zeros((250, 100, 3), dtype=np.uint8) + 150
        )

        eye_plot_layout = QtWidgets.QGridLayout()
        eye_plot_layout.addWidget(self.left_eye_plotter, 1, 1)
        eye_plot_layout.addWidget(self.left_eye_vis_plotter, 2, 1)
        eye_plot_layout.addWidget(self.right_eye_plotter, 1, 2)
        eye_plot_layout.addWidget(self.right_eye_vis_plotter, 2, 2)

        face_plot_layout = QtWidgets.QHBoxLayout()
        face_plot_layout.addWidget(self.face_plotter)
        face_plot_layout.addWidget(self.face_vis_plotter)

        self.record_curr_frame_button.setCheckable(True)
        self.record_curr_frame_button.setChecked(False)
        self.record_curr_frame_button.setShortcut("Ctrl+S")
        
        self.record_state_label = QtWidgets.QLabel()
        self.mouse_pos_raw_label = QtWidgets.QLabel()
        self.mouse_pos_label = QtWidgets.QLabel()
        
        button_N_label_layout = QtWidgets.QHBoxLayout()
        button_N_label_layout.addWidget(self.record_curr_frame_button)
        button_N_label_layout.addWidget(self.record_state_label)
        button_N_label_layout.addWidget(self.mouse_pos_raw_label)
        button_N_label_layout.addWidget(self.mouse_pos_label)

        image_plot_layout = QtWidgets.QVBoxLayout()
        image_plot_layout.addLayout(button_N_label_layout)
        image_plot_layout.addWidget(self.webcam_image_plotter)
        image_plot_layout.addLayout(face_plot_layout)
        image_plot_layout.addLayout(eye_plot_layout)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.three_dimention_visualizer)
        main_layout.addLayout(image_plot_layout)
        
        central_widget = QtWidgets.QLabel()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.show()

    def updateMouseData(self, mouse_data) :
        self.configs.mouse_pos_raw = np.array(mouse_data[0:2])
        self.configs.scroll_val = np.array(mouse_data[2:4])
        self.configs.mouse_pos = self.configs.mouse_pos_raw / self.configs.screen_geometry
        

        self.mouse_pos_raw_label.setText(str(self.configs.mouse_pos_raw))
        self.mouse_pos_label.setText(str(self.configs.mouse_pos))

    def updateFaceData(self, data_dict) :
        """
        update three dimension plot, image plotters.
        """

        shm_queue_idx = data_dict["image_idx"]
        image = self.image_queue[shm_queue_idx].copy()
        image = cv2.flip(image, 1)
        self.webcam_image_plotter.update(image)
        
        if not data_dict["face_detected"] :
            return
        
        face_landmark_array = self.face_queue[shm_queue_idx].copy()
        face_landmark_array[:, 0] = 1 - face_landmark_array[:, 0]
        
        if face_landmark_array[:, :2].min() < 0 :
            return

        self.three_dimention_visualizer.updateFace(face_landmark_array)

        vis_image = image.copy() 
        vis_image = self.two_dimension_visualizer.visualizeFace2D(
            image = vis_image, landmark_array = face_landmark_array
        )

        full_lt_rb = np.array([
            face_landmark_array.min(axis=0),
            face_landmark_array.max(axis=0),
        ])[:, :2] * image.shape[-2:-4:-1]
        full_lt_rb = full_lt_rb.flatten().astype(int)

        left_eye_lt_rb = np.array([
            face_landmark_array[
                np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()
            ].min(axis=0),
            face_landmark_array[
                np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()
            ].max(axis=0),
        ])[:, :2] * image.shape[-2:-4:-1]
        left_eye_lt_rb = left_eye_lt_rb * 2 - left_eye_lt_rb.mean(axis=0)
        left_eye_lt_rb = left_eye_lt_rb.flatten().astype(int)

        right_eye_lt_rb = np.array([
            face_landmark_array[
                np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()
            ].min(axis=0),
            face_landmark_array[
                np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()
            ].max(axis=0),
        ])[:, :2] * image.shape[-2:-4:-1]
        right_eye_lt_rb = right_eye_lt_rb * 2 - right_eye_lt_rb.mean(axis=0)
        right_eye_lt_rb = right_eye_lt_rb.flatten().astype(int)

        image_shape = np.array(image.shape[-2:-4:-1])
        left_iris_center = (face_landmark_array[
            np.array(list(mp_face_mesh.FACEMESH_LEFT_IRIS)).flatten()
        ].mean(axis=0)[:2] * image_shape).astype(int)
        right_iris_center = (face_landmark_array[
            np.array(list(mp_face_mesh.FACEMESH_RIGHT_IRIS)).flatten()
        ].mean(axis=0)[:2] * image_shape).astype(int)

        image = cv2.circle(
            image,
            left_iris_center,
            2,
            (255,255,255)
        )
        image = cv2.circle(
            image,
            right_iris_center,
            2,
            (255,255,255)
        )

        self.face_plotter.update(image[
            full_lt_rb[1]:full_lt_rb[3], full_lt_rb[0]:full_lt_rb[2], :
        ])
        self.face_vis_plotter.update(vis_image[
            full_lt_rb[1]:full_lt_rb[3], full_lt_rb[0]:full_lt_rb[2], :
        ])
        self.left_eye_plotter.update(image[
            left_eye_lt_rb[1]:left_eye_lt_rb[3],
            left_eye_lt_rb[0]:left_eye_lt_rb[2],
            :
        ])
        self.left_eye_vis_plotter.update(vis_image[
            right_eye_lt_rb[1]:right_eye_lt_rb[3],
            right_eye_lt_rb[0]:right_eye_lt_rb[2],
            :
        ])
        self.right_eye_plotter.update(image[
            right_eye_lt_rb[1]:right_eye_lt_rb[3],
            right_eye_lt_rb[0]:right_eye_lt_rb[2],
            :
        ])
        self.right_eye_vis_plotter.update(vis_image[
            right_eye_lt_rb[1]:right_eye_lt_rb[3],
            right_eye_lt_rb[0]:right_eye_lt_rb[2],
            :
        ])

        if self.record_curr_frame_button.isChecked() :
            print(f"<<putting. idx:{shm_queue_idx}")
            self.save_data_queue.put({
                "shm_queue_idx" : shm_queue_idx,
                "mouse_position" : self.configs.mouse_pos
            })

    def saveData(
        self,
        image,
        face_landmark_array,
        mouse_pos
    ) :
        if not self.configs.DATA_DIR_PATH :
            DATA_DIR_NAME = time.strftime("%Y_%m_%d__%H_%M_%S")
            self.configs.DATA_DIR_PATH = os.path.join(self.configs.DATA_ROOT_PATH, DATA_DIR_NAME)
            os.makedirs(self.configs.DATA_DIR_PATH)
        
        timestamp = str(int(time.time()))
        cv2.imwrite(
            os.path.join(self.configs.DATA_DIR_PATH, f"{timestamp}.png"),
            image
        )
        with open(
            os.path.join(self.configs.DATA_DIR_PATH, f"{timestamp}.pkl"), "wb"
        ) as fp :
            pickle.dump(
                {
                    "face_landmark_array" : face_landmark_array,
                    "mouse_position" : mouse_pos
                },
                fp
            )

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key.Key_Space :
            self.record_curr_frame_button.setChecked(True)

    def keyReleaseEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key.Key_Space and not a0.isAutoRepeat() :
            self.record_curr_frame_button.setChecked(False)

if __name__ == "__main__" :
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
