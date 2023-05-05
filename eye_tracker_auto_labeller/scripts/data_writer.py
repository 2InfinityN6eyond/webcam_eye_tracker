import os
import time
import pickle
import json
import cv2
import numpy as np
import multiprocessing
import threading
from multiprocessing import shared_memory

from utils import N_FACE_LANDMARK_FEATURES

class DataWriterWorder(threading.Thread) :
    def __init__(
        self,
        frame_width,
        frame_height,    
        img_shm_name,
        face_shm_name,
        shm_queue_size,
        shm_queue_idx,
        save_dir_path,
        data_queue  : multiprocessing.Queue,
        stop_flag   : multiprocessing.Event,
        save_image_flag
    ) :
        super(DataWriterWorder, self).__init__()
        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.img_shm_name   = img_shm_name
        self.face_shm_name  = face_shm_name
        self.shm_queue_size = shm_queue_size
        self.shm_queue_idx  = shm_queue_idx
        self.save_dir_path  = save_dir_path
        self.data_queue     = data_queue
        self.stop_flag      = stop_flag
        self.save_image_flag = save_image_flag

    def run(self) :
        image_shm = shared_memory.SharedMemory(name=self.img_shm_name)
        image_queue = np.ndarray(
            (self.shm_queue_size, self.frame_height, self.frame_width, 3),
            dtype=np.uint8, buffer = image_shm.buf
        )
        image_array = image_queue[self.shm_queue_idx]

        face_shm = shared_memory.SharedMemory(name = self.face_shm_name)
        face_queue = np.ndarray(
            (self.shm_queue_size, N_FACE_LANDMARK_FEATURES, 3),
            dtype=np.float64, buffer = face_shm.buf
        )
        face_array = face_queue[self.shm_queue_idx]

        while not self.stop_flag.is_set() :
            if not self.data_queue.empty() :
                data = self.data_queue.get()
                mouse_position = data["mouse_position"]

                image = image_array.copy()
                image = cv2.flip(image, 1)
                face_landmark_array = face_array.copy()
                face_landmark_array[:, 0] = 1 - face_landmark_array[:, 0]

                timestamp = str(time.time()).replace('.', '_')

                print(self.shm_queue_idx)

                if self.save_image_flag :
                    cv2.imwrite(
                        os.path.join(self.save_dir_path, f"{timestamp}.png"),
                        image
                    )
                with open(os.path.join(self.save_dir_path, f"{timestamp}.json"), "w") as fp :
                    json.dump(
                        {
                            "mouse_position" : mouse_position.tolist(),
                            "face_landmark_array" : face_landmark_array.tolist(),
                        },
                        fp, indent = 2
                )
        image_shm.close()
        face_shm.close()

class DataWriter(multiprocessing.Process) :
    def __init__(
        self,
        frame_width     : int,
        frame_height    : int,
        data_root_path  : str,
        img_shm_name    : str,
        face_shm_name   : str,
        shm_queue_size  : int,
        save_data_queue : multiprocessing.Queue,
        stop_flag       : multiprocessing.Event,
        save_every_data_flag : bool = False,
        save_image_flag : bool = True,
    ) :
        super(DataWriter, self).__init__()
        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.data_root_path = data_root_path
        self.img_shm_name   = img_shm_name
        self.face_shm_name  = face_shm_name
        self.shm_queue_size = shm_queue_size
        self.save_data_queue= save_data_queue
        self.stop_flag      = stop_flag
        self.save_every_data_flag = save_every_data_flag
        self.save_image_flag = save_image_flag

        self.save_dir_path = os.path.join(
            self.data_root_path,
            time.strftime("%Y_%m_%d__%H_%M_%S")
        )
        os.makedirs(self.save_dir_path)

    def run(self) :
        self.data_writer_worker_queue_list = list(map(
            lambda idx : multiprocessing.Queue(),
            range(self.shm_queue_size)
        ))
        self.data_writer_worker_list = list(map(
            lambda idx, data_queue : DataWriterWorder(
                frame_width     = self.frame_width,
                frame_height    = self.frame_height,
                img_shm_name    = self.img_shm_name,
                face_shm_name   = self.face_shm_name,
                shm_queue_size  = self.shm_queue_size,
                shm_queue_idx   = idx,
                save_dir_path   = self.save_dir_path,
                data_queue      = data_queue,
                stop_flag       = self.stop_flag,
                save_image_flag = self.save_image_flag
            ),
            range(self.shm_queue_size),
            self.data_writer_worker_queue_list
        ))
        list(map(lambda worker : worker.start(), self.data_writer_worker_list))

        while not self.stop_flag.is_set() :
            if not self.save_data_queue.empty() :
                data = self.save_data_queue.get()
                shm_idx = data["shm_queue_idx"]
                
                if self.save_every_data_flag :
                    self.data_writer_worker_queue_list[shm_idx].put(data)
                    print(f"recieved. idx:{shm_idx}");  continue
                elif shm_idx % self.shm_queue_size == 0 :
                    self.data_writer_worker_queue_list[shm_idx].put(data)
                    print(f"recieved. idx:{shm_idx}");  continue

                

        if not os.listdir(self.save_dir_path) :
            os.removedirs(self.save_dir_path)
        
        print("joining workers..")
        list(map(lambda worker : worker.join(), self.data_writer_worker_list))
        print("worker joined")
