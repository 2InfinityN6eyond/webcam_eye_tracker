import os
import numpy as np
import multiprocessing
from multiprocessing import shared_memory

import cv2
import mediapipe as mp

from utils import N_FACE_LANDMARK_FEATURES

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class FaceLandmarkProcessor(multiprocessing.Process) :
    def __init__(
            self,
            frame_width     :int,
            frame_height    :int,
            img_shm_name    :str,
            face_shm_name   :str,
            shm_queue_size  :int,
            shm_idx_queue   :multiprocessing.Queue,
            data_queue      :multiprocessing.Queue,
            stop_flag       :multiprocessing.Event,
            max_num_faces   :int  = 1,
        ) :
        super(FaceLandmarkProcessor, self).__init__()
        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.img_shm_name   = img_shm_name
        self.face_shm_name  = face_shm_name
        self.shm_queue_size = shm_queue_size
        self.shm_idx_queue  = shm_idx_queue
        self.data_queue     = data_queue
        self.stop_flag      = stop_flag
        self.max_num_faces  = max_num_faces

    def run(self) :
        image_shm = shared_memory.SharedMemory(name=self.img_shm_name)
        image_queue = np.ndarray(
            (self.shm_queue_size, self.frame_height, self.frame_width, 3),
            dtype=np.uint8, buffer = image_shm.buf
        )

        face_shm = shared_memory.SharedMemory(name = self.face_shm_name)
        face_queue = np.ndarray(
            (self.shm_queue_size, N_FACE_LANDMARK_FEATURES, 3),
            dtype=np.float64, buffer = face_shm.buf
        )

        with mp_face_mesh.FaceMesh(
            max_num_faces            = self.max_num_faces,
            refine_landmarks         = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5
        ) as face_mesh :
            while not self.stop_flag.is_set() :
                if self.shm_idx_queue.empty() :
                    continue
                shm_queue_idx = self.shm_idx_queue.get()
                image = image_queue[shm_queue_idx].copy()
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                data_dict = {"image_idx": shm_queue_idx, "face_detected": False}
                
                if results.multi_face_landmarks :
                    face_landmark_array = np.array(
                        list(map(
                            lambda kp : [kp.x, kp.y, kp.z],
                            results.multi_face_landmarks[0].landmark
                        )),
                        dtype = np.float64
                    )

                    face_queue[shm_queue_idx, :, :] = face_landmark_array[:, :]
                    data_dict["face_detected"] = True
                self.data_queue.put(data_dict)
            image_shm.close()
            face_shm.close()