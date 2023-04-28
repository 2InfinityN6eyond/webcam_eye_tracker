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
