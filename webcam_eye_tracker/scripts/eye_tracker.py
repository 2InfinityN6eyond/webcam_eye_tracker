import os
import numpy as np
import multiprocessing
from multiprocessing import shared_memory
import cv2
import mediapipe as mp

from pynput.mouse import Button, Controller

import torch

from utils import (
    N_FACE_LANDMARK_FEATURES, FACE_OVAL_LANDMARK_IDX_LIST,
    LEFT_EYE_LANDMARK_IDX_LIST, LEFT_IRIS_LANDMARK_IDX_LIST,
    RIGHT_EYE_LANDMARK_IDX_LIST, RIGHT_IRIS_LANDMARK_IDX_LIST
)
from eye_tracker_net import EyeTrackerNet

class EyeTracker(multiprocessing.Process) :
    def __init__(
            self,
            screen_width    :int,
            screen_height   :int,
            face_shm_name   :str,
            shm_queue_size  :int,
            shm_idx_queue   :multiprocessing.Queue,
            stop_flag       :multiprocessing.Event,
            #model_path      :str = "../../eye_tracker_model/checkpoints/model_20230428_105603_199"
            model_path      :str = "/Volumes/HJP/PROJECTS/webcam_eye_tracker/eye_tracker_model/checkpoints/model_20230504_215350_198"
        ) :
        super(EyeTracker, self).__init__()
        self.screen_width   = screen_width
        self.screen_height  = screen_height
        self.screen_geomotry= np.array([screen_width, screen_height])
        self.face_shm_name  = face_shm_name
        self.shm_queue_size = shm_queue_size
        self.shm_idx_queue  = shm_idx_queue
        self.stop_flag      = stop_flag
        self.model_path     = model_path

    def run(self) :
        model = EyeTrackerNet()
        model.load_state_dict(torch.load(self.model_path))

        face_shm = shared_memory.SharedMemory(name = self.face_shm_name)
        face_queue = np.ndarray(
            (self.shm_queue_size, N_FACE_LANDMARK_FEATURES, 3),
            dtype=np.float64, buffer = face_shm.buf
        )

        mouse_controller = Controller()
    
        while not self.stop_flag.is_set() :
            if self.shm_idx_queue.empty() :
                continue
            shm_queue_idx = self.shm_idx_queue.get()
            
            face_landmark_array = face_queue[shm_queue_idx].copy()
            face_landmark_array[:, 0] = 1 - face_landmark_array[:, 0]

            eye_tracker_input = torch.Tensor(face_landmark_array[
                FACE_OVAL_LANDMARK_IDX_LIST +
                LEFT_EYE_LANDMARK_IDX_LIST  + LEFT_IRIS_LANDMARK_IDX_LIST +
                RIGHT_EYE_LANDMARK_IDX_LIST + RIGHT_IRIS_LANDMARK_IDX_LIST
            ]).flatten()

            pred = model(eye_tracker_input).detach().numpy()
            
            mouse_position = (pred * self.screen_geomotry).astype(int)
            print(mouse_position)
            mouse_controller.position = mouse_position

        face_shm.close()