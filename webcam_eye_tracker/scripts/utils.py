import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

POSE_TRAJECTORY_PATH = []
HAND_TRAJECTORY_PATH = []
FACE_TRAJECTORY_PATH = []

N_FACE_LANDMARK_FEATURES = 478

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
