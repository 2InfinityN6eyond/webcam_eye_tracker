import os
import json
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader

DATA_ROOT_PATH = "../eye_tracker_auto_labeller/data"

FACE_OVAL_LANDMARK_IDX_LIST = [
    162,  21,  54, 103,  67, 109,  10, 338, 297, 332, 284, 251, 389,
    356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172,  58, 132,  93, 234, 127, 162,
]
LEFT_EYE_LANDMARK_IDX_LIST = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 
    398, 384, 385, 386, 387, 388, 466, 263
]
LEFT_IRIS_LANDMARK_IDX_LIST = [475, 476, 477, 474, 475]
RIGHT_EYE_LANDMARK_IDX_LIST = [
     33,   7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246, 33
]
RIGHT_IRIS_LANDMARK_IDX_LIST = [471, 472, 469, 470, 471]

class EyeTrackerDataset(Dataset) :
    def __init__(
        self,
        data_root_path = DATA_ROOT_PATH,
        face_oval_landmark_idx_list     = FACE_OVAL_LANDMARK_IDX_LIST,
        left_eye_landmark_idx_list      = LEFT_EYE_LANDMARK_IDX_LIST,
        left_iris_landmark_idx_list     = LEFT_IRIS_LANDMARK_IDX_LIST,
        right_eye_landmark_idx_list     = RIGHT_EYE_LANDMARK_IDX_LIST,
        right_iris_landmark_idx_list    = RIGHT_IRIS_LANDMARK_IDX_LIST
    ) :
        super(EyeTrackerDataset, self).__init__()
        self.DATA_ROOT_PATH = data_root_path
        self.face_oval_landmark_idx_list    = face_oval_landmark_idx_list
        self.left_eye_landmark_idx_list     = left_eye_landmark_idx_list
        self.left_iris_landmark_idx_list    = left_iris_landmark_idx_list
        self.right_eye_landmark_idx_list    = right_eye_landmark_idx_list
        self.right_iris_landmark_idx_list   = right_iris_landmark_idx_list
        self.file_path_list = sorted(glob(
            f"{DATA_ROOT_PATH}/*/*.json"
        ))
    def __len__(self) :
        return len(self.file_path_list)
    
    def __getitem__(self, index):
        with open(self.file_path_list[index], "r") as fp :
            data = json.load(fp)
        mouse_position = torch.Tensor(data["mouse_position"])
        face_landmark_array = torch.Tensor(data["face_landmark_array"])[
            self.face_oval_landmark_idx_list +
            self.left_eye_landmark_idx_list +
            self.left_iris_landmark_idx_list +
            self.right_eye_landmark_idx_list +
            self.right_iris_landmark_idx_list
        ]
        return face_landmark_array.flatten(), mouse_position
   
