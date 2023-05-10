import sys
import numpy as np
import mediapipe as mp
import cv2

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

mp_face_mesh = mp.solutions.face_mesh

OVAL_UPPER_HALF = [361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132]
OVAL_LOWER_HALF = [234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454]
OVAL_LEFT_HALF =  [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
OVAL_RIGHT_HALF = [148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def edge_list_2_path(edge_list) :
    tesel = edge_list
    change_occured = True
    while change_occured :
        change_occured = False
        for idx in range(len(tesel)) :
            target_edge = tesel[idx]
            inner_changed = False
            for e in tesel :
                if e != target_edge and len(e) < 3 :
                    edge = e
                    if target_edge[-1] == edge[0] :
                        target_edge.append(edge[-1])
                        tesel.remove(edge)
                        change_occured = True
                        inner_changed = True
                        break
                    if target_edge[0] == edge[-1] :
                        target_edge.insert(0, edge[0])
                        tesel.remove(edge)
                        change_occured = True
                        inner_changed  = True
                        break
            if inner_changed :
                break
    change_occured = True
    while change_occured :
        change_occured = False
        for idx in range(len(tesel)) :
            source_path = tesel[idx]
            inner_changed = False
            for target_path in tesel :
                if target_path == source_path :
                    continue
                if source_path[-1] == target_path[-1] :
                    target_path.reverse()
                    source_path += target_path[1:]
                    tesel.remove(target_path)
                    change_occured = True
                    inner_changed = True
                    break
                if source_path[0] == target_path[0] :
                    source_path.reverse()
                    target_path += source_path[1:]
                    tesel.remove(source_path)
                    change_occured = True
                    inner_changed = True
                    break
            if inner_changed :
                break
    return tesel


FACE_TESSELATION_PATH_LIST = edge_list_2_path(np.array(list(
    mp_face_mesh.FACEMESH_TESSELATION
)).tolist())
FACE_OVAL_PATH_LIST = edge_list_2_path(np.array(list(
    mp_face_mesh.FACEMESH_FACE_OVAL
)).tolist())
FACE_LIPS_PATH_LIST = edge_list_2_path(np.array(list(
    mp_face_mesh.FACEMESH_LIPS
)).tolist())
FACE_LEFT_EYEBROW_PATH_LIST = edge_list_2_path(np.array(list(
    mp_face_mesh.FACEMESH_LEFT_EYEBROW
)).tolist())
FACE_LEFT_EYE_PATH_LIST = edge_list_2_path(np.array(list(
    mp_face_mesh.FACEMESH_LEFT_EYE
)).tolist())
FACE_LEFT_IRIS_PATH_LIST = edge_list_2_path(np.array(list(
    mp_face_mesh.FACEMESH_LEFT_IRIS
)).tolist())
FACE_RIGHT_EYEBROW_PATH_LIST = edge_list_2_path(np.array(list(
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW
)).tolist())
FACE_RIGHT_EYE_PATH_LIST = edge_list_2_path(np.array(list(
    mp_face_mesh.FACEMESH_RIGHT_EYE
)).tolist())
FACE_RIGHT_IRIS_PATH_LIST = edge_list_2_path(np.array(list(
    mp_face_mesh.FACEMESH_RIGHT_IRIS
)).tolist())

class TwoDimensionVisualizer() :
    def __init__(
        self
    ) :
        pass

    def visualizeFace2D(
        self,
        image,
        landmark_array
    ) :
        
        face_landmark_array = landmark_array.copy()
        ver_mean_diff = np.mean(face_landmark_array[OVAL_LOWER_HALF], axis=0) - np.mean(face_landmark_array[OVAL_UPPER_HALF], axis=0)
        hor_mean_diff = np.mean(face_landmark_array[OVAL_RIGHT_HALF], axis=0) - np.mean(face_landmark_array[OVAL_LEFT_HALF],  axis=0)
        normal_vec = np.cross(hor_mean_diff, ver_mean_diff)
        '''print(
            "{0:.4f} {1:.4f} {2:.4f} {3}".format(
                np.dot(ver_mean_diff, hor_mean_diff),
                np.sqrt(np.sum(ver_mean_diff * ver_mean_diff)),
                np.sqrt(np.sum(hor_mean_diff * hor_mean_diff)),
                normal_vec
            )
        )'''

        landmark_2d_array = landmark_array[:, :2].copy()
        landmark_2d_array *= np.array([
            image.shape[1], image.shape[0]
        ])
        landmark_2d_array = landmark_2d_array.astype(int)
        
        list(map(
            lambda idx_list : cv2.polylines(
                image,
                [landmark_2d_array[idx_list]],
                isClosed=False,
                color=(255,255,255, 10),
                thickness=1
            ),
            FACE_TESSELATION_PATH_LIST +
            FACE_LEFT_EYE_PATH_LIST +
            FACE_LEFT_IRIS_PATH_LIST +
            FACE_RIGHT_EYE_PATH_LIST +
            FACE_RIGHT_IRIS_PATH_LIST
        ))
        list(map(
            lambda idx_list : cv2.polylines(
                image,
                [landmark_2d_array[idx_list]],
                isClosed=False,
                color=(255,255,255, 10),
                thickness=2
            ),
            FACE_LIPS_PATH_LIST + 
            FACE_OVAL_PATH_LIST + 
            FACE_LEFT_EYEBROW_PATH_LIST +
            FACE_RIGHT_EYEBROW_PATH_LIST
        ))
        return image

class ThreeDimensionVisualizer(gl.GLViewWidget) :
    def __init__(self) -> None :
        super().__init__()

        gx = gl.GLGridItem(color=pg.mkColor((100, 50, 50)))
        gx.rotate(90, 0, 1, 0)
        self.addItem(gx)
        gy = gl.GLGridItem(color=pg.mkColor((50, 100, 50)))
        gy.rotate(90, 1, 0, 0)
        self.addItem(gy)
        gz = gl.GLGridItem(color=pg.mkColor((50, 50, 100)))
        self.addItem(gz)

        self.setCameraParams(elevation = -90, azimuth = -90)

        self.left_hand_line_list = []
        self.right_hand_line_list = []
        self.face_line_list = []
        self.pose_line_list = []

    def updateFace(
        self,
        landmark_array,
        set_mean_as_origin = True,
        always_place_center = False,
    ) : 
        landmark_array = landmark_array.copy()
        for line_item in self.face_line_list :
            self.removeItem(line_item)
        if set_mean_as_origin :
            landmark_array -= np.array([0.5, 0.5, 0])
        elif always_place_center :
            landmark_array -= landmark_array.mean(axis=0)
        self.face_line_list = list(map(
            lambda face_path : gl.GLLinePlotItem(
                pos = landmark_array[face_path],
                color = pg.mkColor((255, 0, 0)), width = 2,
                antialias = True
            ),            
            FACE_TESSELATION_PATH_LIST + 
            FACE_LIPS_PATH_LIST + 
            FACE_OVAL_PATH_LIST + 
            FACE_LEFT_EYEBROW_PATH_LIST +
            FACE_LEFT_EYE_PATH_LIST +
            FACE_LEFT_IRIS_PATH_LIST +
            FACE_RIGHT_EYEBROW_PATH_LIST + 
            FACE_RIGHT_EYE_PATH_LIST +
            FACE_RIGHT_IRIS_PATH_LIST
        ))

        ver_mean_diff = np.mean(landmark_array[OVAL_LOWER_HALF], axis=0) - np.mean(landmark_array[OVAL_UPPER_HALF], axis=0)
        hor_mean_diff = np.mean(landmark_array[OVAL_RIGHT_HALF], axis=0) - np.mean(landmark_array[OVAL_LEFT_HALF],  axis=0)
        normal_vec = np.cross(hor_mean_diff, ver_mean_diff)
        normal_line_item = gl.GLLinePlotItem(
            pos = np.array([ [0, 0, 0], normal_vec ]) + landmark_array[FACE_OVAL_PATH_LIST[0]].mean(axis=0),
            color = pg.mkColor((255, 255, 255)), width = 10,
            antialias = True
        )
        self.face_line_list.append(normal_line_item)

        eye_normal_line_item = gl.GLLinePlotItem(
            pos = np.array([
                [0, 0, 0], normal_vec 
            ]) + landmark_array[FACE_LEFT_EYE_PATH_LIST[0] + FACE_RIGHT_EYE_PATH_LIST[0]].mean(axis=0),
            color = pg.mkColor((255, 255, 255)), width = 10,
            antialias = True
        )
        self.face_line_list.append(eye_normal_line_item)

        pitch   = np.arctan2(normal_vec[1], -normal_vec[2])
        yaw     = np.arctan2(normal_vec[0], -normal_vec[2])
        #roll    = np.rad2deg()

        test_vector = np.array([np.tan(yaw), np.tan(pitch), -1])

        self.face_line_list.append(
            gl.GLLinePlotItem(
                pos = np.array([
                    [0, 0, 0], test_vector 
                ]) + landmark_array[FACE_LEFT_EYE_PATH_LIST[0] + FACE_RIGHT_EYE_PATH_LIST[0]].mean(axis=0),
                color = pg.mkColor((100, 255, 255)), width = 1,
                antialias = True
            )
        )

        for item in self.face_line_list :
            if item :
                self.addItem(item)

        print(
            "{0:.4f} {1:.4f} {2} {3}".format(
                np.rad2deg(yaw), np.rad2deg(pitch), landmark_array[FACE_LEFT_IRIS_PATH_LIST].mean(axis=0), landmark_array[FACE_RIGHT_EYE_PATH_LIST].mean(axis=0) # normal_vec
            )
        )