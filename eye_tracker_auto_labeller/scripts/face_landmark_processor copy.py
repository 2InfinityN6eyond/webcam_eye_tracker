import numpy as np
import multiprocessing
from multiprocessing import shared_memory

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class FaceLandmarkProcessor(multiprocessing.Process) :
    def __init__(
            self,
            frame_width     :int,
            frame_height    :int,
            shm_name        :str,
            img_queue_size  :int,
            img_idx_queue   :multiprocessing.Queue,
            data_queue      :multiprocessing.Queue,
            stop_flag       :multiprocessing.Event,
            show            :bool = True
        ) :
        super(FaceLandmarkProcessor, self).__init__()
        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.shm_name       = shm_name
        self.img_queue_size = img_queue_size
        self.img_idx_queue  = img_idx_queue
        self.data_queue     = data_queue
        self.stop_flag      = stop_flag
        self.show = show

    def run(self) :

        shm = shared_memory.SharedMemory(name=self.shm_name)
        image_queue = np.ndarray(
            (self.img_queue_size, self.frame_height, self.frame_width, 3),
            dtype=np.uint8, buffer = shm.buf
        )

        cv2.namedWindow("left_eye", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname="left_eye", width = 800, height=800)
        cv2.namedWindow("right_eye", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname="right_eye", width = 800, height=800)
        cv2.moveWindow(winname="right_eye", x = 0, y = 800)

        cv2.namedWindow("face", cv2.WINDOW_NORMAL)
        cv2.moveWindow(winname="face", x = 800, y = 0)
        
        cv2.namedWindow("face2", cv2.WINDOW_NORMAL)
        cv2.moveWindow(winname="face2", x = 800, y = 0)

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh :
            while not self.stop_flag.is_set() :
                image_idx = self.img_idx_queue.get()

                image = image_queue[image_idx]
                image = cv2.flip(image, 1)
                
                vis_image_1 = image.copy()
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                data_dict = {
                    "image_idx" : image_idx
                }
                if results.multi_face_landmarks :
                    face_landmarks = results.multi_face_landmarks[0]

                    mp_drawing.draw_landmarks(
                        image=vis_image_1,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=vis_image_1,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=vis_image_1,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                    )

                    face_landmark_list = list(map(
                        lambda kp : [kp.x, kp.y, kp.z],
                        face_landmarks.landmark
                    ))
                    data_dict["face"] = face_landmark_list
                    self.data_queue.put(data_dict)
            
                if self.show :
                    cv2.imshow("image", vis_image_1)
                    
                    if results.multi_face_landmarks :
                        face_landmarks_array = np.array(face_landmark_list)
                        full_lt_rb = np.array([
                            face_landmarks_array.min(axis=0),
                            face_landmarks_array.max(axis=0),
                        ])[:, :2] * image.shape[-2:-4:-1]
                        full_lt_rb = full_lt_rb.flatten().astype(int)

                        left_eye_lt_rb = np.array([
                            face_landmarks_array[
                                np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()
                            ].min(axis=0),
                            face_landmarks_array[
                                np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()
                            ].max(axis=0),
                        ])[:, :2] * image.shape[-2:-4:-1]
                        left_eye_lt_rb = left_eye_lt_rb * 2 - left_eye_lt_rb.mean(axis=0)
                        left_eye_lt_rb = left_eye_lt_rb.flatten().astype(int)

                        right_eye_lt_rb = np.array([
                            face_landmarks_array[
                                np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()
                            ].min(axis=0),
                            face_landmarks_array[
                                np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()
                            ].max(axis=0),
                        ])[:, :2] * image.shape[-2:-4:-1]
                        right_eye_lt_rb = right_eye_lt_rb * 2 - right_eye_lt_rb.mean(axis=0)
                        right_eye_lt_rb = right_eye_lt_rb.flatten().astype(int)


                        image_shape = np.array(image.shape[-2:-4:-1])
                        left_iris_center = (face_landmarks_array[
                            np.array(list(mp_face_mesh.FACEMESH_LEFT_IRIS)).flatten()
                        ].mean(axis=0)[:2] * image_shape).astype(int)
                        right_iris_center = (face_landmarks_array[
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
                        vis_image_1 = cv2.circle(
                            vis_image_1,
                            left_iris_center,
                            2,
                            (255,255,255)
                        )
                        vis_image_1 = cv2.circle(
                            vis_image_1,
                            right_iris_center,
                            2,
                            (255,255,255)
                        )

                        left_eye = np.vstack([
                            image[left_eye_lt_rb[1]:left_eye_lt_rb[3], left_eye_lt_rb[0]:left_eye_lt_rb[2], : ],
                            vis_image_1[left_eye_lt_rb[1]:left_eye_lt_rb[3], left_eye_lt_rb[0]:left_eye_lt_rb[2], : ]
                        ])
                        right_eye = np.vstack([
                            image[right_eye_lt_rb[1]:right_eye_lt_rb[3], right_eye_lt_rb[0]:right_eye_lt_rb[2], :],
                            vis_image_1[right_eye_lt_rb[1]:right_eye_lt_rb[3], right_eye_lt_rb[0]:right_eye_lt_rb[2], :]
                        ])
                            
                        cv2.imshow("left_eye", left_eye)
                        cv2.imshow("right_eye", right_eye)

                        cv2.imshow("face", vis_image_1[
                            full_lt_rb[1]:full_lt_rb[3], full_lt_rb[0]:full_lt_rb[2], :
                        ])

                        cv2.imshow("face2", image[
                            full_lt_rb[1]:full_lt_rb[3], full_lt_rb[0]:full_lt_rb[2], :
                        ])


                    in_key = cv2.waitKey(1)
                    if in_key in [ord("q"), 27] :
                        break

            shm.close()

if __name__ == "__main__" :
    queue = multiprocessing.Queue
    holistic_reader = FaceReader(queue)