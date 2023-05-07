import os
import json
import numpy as np
from glob import glob
from PIL import Image
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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

@dataclass
class EyeTrackerData :
    mouse_position  : torch.Tensor
    image           : torch.Tensor
    left_eye_image  : torch.Tensor
    right_eye_image : torch.Tensor
    
    face_landmark_array         : torch.Tensor
    face_oval_landmark_array    : torch.Tensor
    left_eye_landmark_array     : torch.Tensor
    right_eye_landmark_array    : torch.Tensor
    left_iris_landmark_array    : torch.Tensor
    right_iris_landmark_arrya   : torch.Tensor



class EyeTrackerDataset(Dataset) :
    def __init__(
        self,
        data_root_path = DATA_ROOT_PATH,
        face_oval_landmark_idx_list     = FACE_OVAL_LANDMARK_IDX_LIST,
        left_eye_landmark_idx_list      = LEFT_EYE_LANDMARK_IDX_LIST,
        left_iris_landmark_idx_list     = LEFT_IRIS_LANDMARK_IDX_LIST,
        right_eye_landmark_idx_list     = RIGHT_EYE_LANDMARK_IDX_LIST,
        right_iris_landmark_idx_list    = RIGHT_IRIS_LANDMARK_IDX_LIST,
        return_landmark = True,
        return_image = False
    ) :
        super(EyeTrackerDataset, self).__init__()
        self.DATA_ROOT_PATH = data_root_path
        self.face_oval_landmark_idx_list    = face_oval_landmark_idx_list
        self.left_eye_landmark_idx_list     = left_eye_landmark_idx_list
        self.left_iris_landmark_idx_list    = left_iris_landmark_idx_list
        self.right_eye_landmark_idx_list    = right_eye_landmark_idx_list
        self.right_iris_landmark_idx_list   = right_iris_landmark_idx_list
        self.return_landmark = return_landmark
        self.return_image = return_image

        self.label_path_list = sorted(glob(
            f"{self.DATA_ROOT_PATH}/*/*.json"
        ))
        self.image_path_list = sorted(glob(
            f"{self.DATA_ROOT_PATH}/*/*.png"
        ))

        # there might exists case where eiter image or label does not exist
        # ex) a.json exist but a.png doesn't exist or vice-versa.
        # filter this cases to ensure every data are image-label pair.
        self.label_path_list = list(filter(
            lambda file_name : file_name.replace("json", "png") in self.image_path_list,
            self.label_path_list
        ))
        self.image_path_list = list(filter(
            lambda file_name : file_name.replace("png", "json") in self.label_path_list,
            self.image_path_list
        ))
        assert len(self.label_path_list) == len(self.image_path_list)

        self.to_torch = transforms.ToTensor()

    def __len__(self) :
        return len(self.label_path_list)
    
    def __getitem__(self, index):
        with open(self.label_path_list[index], "r") as fp :
            label_data = json.load(fp)
        mouse_position = torch.Tensor(label_data["mouse_position"])
        data = {
            "mouse_position" : mouse_position
        }
        if self.return_landmark :
            face_landmark_array = torch.Tensor(label_data["face_landmark_array"])
            face_oval_landmark_array  = face_landmark_array[self.face_oval_landmark_idx_list]
            left_eye_landmark_array   = face_landmark_array[self.left_eye_landmark_idx_list]
            right_eye_landmark_array  = face_landmark_array[self.right_eye_landmark_idx_list]
            left_iris_landmark_array  = face_landmark_array[self.left_iris_landmark_idx_list]
            right_iris_landmark_array = face_landmark_array[self.right_iris_landmark_idx_list]

            data["face_landmark_array"] = face_landmark_array
            data["face_oval_landmark_array"] = face_oval_landmark_array
            data["left_eye_landmark_array"] = left_eye_landmark_array
            data["right_eye_landmark_array"] = right_eye_landmark_array
            data["left_iris_landmark_array"] = left_iris_landmark_array
            data["right_iris_landmark_array"] = right_iris_landmark_array

        if self.return_image :
            image = Image.open(self.image_path_list[index])
            left_eye_lt_rb = np.array([
                left_eye_landmark_array.numpy().min(axis=0),
                left_eye_landmark_array.numpy().max(axis=0),
            ])[:, :2] * np.array([image.width, image.height])
            #left_eye_lt_rb = left_eye_lt_rb * 2 - left_eye_lt_rb.mean(axis=0)
            left_eye_lt_rb = left_eye_lt_rb.flatten().astype(int)

            right_eye_lt_rb = np.array([
                right_eye_landmark_array.numpy().min(axis=0),
                right_eye_landmark_array.numpy().max(axis=0),
            ])[:, :2] * np.array([image.width, image.height])
            #right_eye_lt_rb = right_eye_lt_rb * 2 - right_eye_lt_rb.mean(axis=0)
            right_eye_lt_rb = right_eye_lt_rb.flatten().astype(int)

            left_eye_image  = self.to_torch(image.crop(left_eye_lt_rb))
            right_eye_image = self.to_torch(image.crop(right_eye_lt_rb))
            
            data["image"] =  self.to_torch(image)
            data["left_eye_image"] =  left_eye_image
            data["right_eye_image"] =  right_eye_image
        
        return data
   
if __name__ == "__main__" :
    eye_tracker_dataset = EyeTrackerDataset()
    print("len :", len(eye_tracker_dataset))
    data = eye_tracker_dataset[0]