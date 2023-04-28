import os
import sys
import platform
import numpy as np
import cv2
import time
import multiprocessing
from multiprocessing import shared_memory
import argparse

from pynput import mouse

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from utils import N_FACE_LANDMARK_FEATURES
from camera_reader import CameraReader
from face_landmark_processor import FaceLandmarkProcessor
from mediapipe_visualizer import ThreeDimensionVisualizer
from main_window import MainWindow
from data_writer import DataWriter


class FakeWriter(multiprocessing.Process) :
    def __init__(self, queue, stop_flag) :
        super(FakeWriter, self).__init__()
        self.queue = queue
        self.stop_flag = stop_flag
    
    def run(self) :
        while not self.stop_flag.is_set() :
            if not self.queue.empty() :
                data = self.queue.get()

                print(data)

class DataBridge(QtCore.QThread) :
    landmark_acquired = QtCore.pyqtSignal(dict)
    mouse_changed = QtCore.pyqtSignal(list)
    
    def __init__(self, stop_flag, data_queue) :
        super(DataBridge, self).__init__()
        self.stop_flag = stop_flag
        self.data_queue = data_queue

    def run(self) :
        listener = mouse.Listener(
            on_move = lambda x, y : self.mouse_changed.emit(
                [int(x), int(y), 0, 0, None, None]
            ),
            on_click = lambda x, y, button, pressed : self.mouse_changed.emit(
                [int(x), int(y), 0, 0, button, pressed]
            ),
            on_scroll = lambda x, y, dx, dy : self.mouse_changed.emit(
                [int(x), int(y), int(dx), int(dy), None, None]
            )
        )
        listener.start()

        while not self.stop_flag.is_set() :
            self.landmark_acquired.emit(
                self.data_queue.get()
            )
        
        listener.stop()
        listener.join()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_idx",         default=0, type=int)
    parser.add_argument("--image_queue_size",   default=4, type=int)
    parser.add_argument("--image_width",        default=1552, type=int)
    parser.add_argument("--image_height",       default=1552, type=int)
    args = parser.parse_args()

    PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    camera_idx = args.camera_idx

    SYSTEM_NAME = platform.system()
    if SYSTEM_NAME == "Windows" :
        VID_CAP_FLAG = cv2.CAP_DSHOW
    if SYSTEM_NAME == "Darwin" :
        VID_CAP_FLAG = None

    # open webcam first and get image size.
    # image shape should be square
    cap = cv2.VideoCapture(camera_idx, VID_CAP_FLAG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.image_height)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("webcam index :", camera_idx)
    print("webcam image size set to {}X{}".format(
        frame_width, frame_height
    ))

    # initialize shared memory for storing camera frames
    image_queue_shm = shared_memory.SharedMemory(
        create = True,
        size   = args.image_queue_size * frame_width * frame_height * 3 
    )
    face_landmark_queue_shm = shared_memory.SharedMemory(
        create  = True,
        size    = args.image_queue_size * N_FACE_LANDMARK_FEATURES * 3 * 8
    )

    # flag for stop program. chile processes stop if stop_flag set.
    stop_flag = multiprocessing.Event()
    # queue accross processes for sending index where image is stored in shm
    shm_idx_queue = multiprocessing.Queue()
    # queue accross processes for sending face mesh data
    data_queue = multiprocessing.Queue()
    save_data_queue = multiprocessing.Queue()

    # initialize pyqt app
    app = QtWidgets.QApplication(sys.argv)
    screen_geometry = app.primaryScreen().geometry()
    main_window = MainWindow(
        proj_root_dir   = PROJECT_ROOT_PATH, 
        screen_width    = screen_geometry.width(),
        screen_height   = screen_geometry.height(),
        frame_width     = frame_width,
        frame_height    = frame_height,
        img_shm_name    = image_queue_shm.name,
        face_shm_name   = face_landmark_queue_shm.name,
        shm_queue_size  = args.image_queue_size,
        save_data_queue = save_data_queue
    )
    main_window.show()

    camera_reader = CameraReader(
        camera_idx      = camera_idx,
        frame_width     = frame_width,
        frame_height    = frame_height,
        img_shm_name    = image_queue_shm.name,
        img_queue_size  = args.image_queue_size,
        img_idx_queue   = shm_idx_queue,
        stop_flag       = stop_flag,
    )
    face_landmark_processor = FaceLandmarkProcessor(
        frame_width     = frame_width,
        frame_height    = frame_height,
        img_shm_name    = image_queue_shm.name,
        face_shm_name   = face_landmark_queue_shm.name,
        shm_queue_size  = args.image_queue_size,
        shm_idx_queue   = shm_idx_queue,
        data_queue      = data_queue,
        stop_flag       = stop_flag,
    )
    '''
    data_writer = FakeWriter(save_data_queue, stop_flag)
    '''
    data_writer = DataWriter(
        frame_width     = frame_width,
        frame_height    = frame_height,
        data_root_path  = os.path.join(PROJECT_ROOT_PATH, "data"),
        img_shm_name    = image_queue_shm.name,
        face_shm_name   = face_landmark_queue_shm.name,
        shm_queue_size  = args.image_queue_size,
        save_data_queue = save_data_queue,
        stop_flag       = stop_flag
    )
    

    data_bridge = DataBridge(
        stop_flag = stop_flag,
        data_queue = data_queue
    )
    
    data_bridge.landmark_acquired.connect(
        lambda ladmark_dict : main_window.updateFaceData(ladmark_dict)
    )
    data_bridge.mouse_changed.connect(
        lambda mouse_data : main_window.updateMouseData(mouse_data)
    )

    data_writer.start()
    data_bridge.start()
    face_landmark_processor.start()
    camera_reader.start()

    app.exec()

    stop_flag.set()
    camera_reader.join()
    face_landmark_processor.join()

    print("face_landmark_joined")

    time.sleep(2)
    data_writer.join()

    print("data_writer joined")

    image_queue_shm.close()
    image_queue_shm.unlink()

    face_landmark_queue_shm.close()
    face_landmark_queue_shm.unlink()
    sys.exit()