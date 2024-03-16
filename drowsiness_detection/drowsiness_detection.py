import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist


class DrowsinessDetector:
    def __init__(self, shape_predictor_path: str):

        self.ratio_threshold = 0.3

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)

        self.l_eye_idx = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.r_eye_idx = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        pass

    def detect_drowsiness(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces_bbox = self.detect_faces(gray)

        ratio = 1
        if faces_bbox:
            face_bbox = faces_bbox[0]  # TODO: select nearest face
            face_landmark = self.predict_face_landmark(gray, face_bbox)

            left_eye_lm, right_eye_lm = self.get_eye_landmarks(face_landmark)
            ratio = self.drowsiness_ratio(left_eye_lm, right_eye_lm)

        predict = ratio < self.ratio_threshold

        return predict

    def detect_faces(self, image: np.ndarray) -> list:
        faces = self.detector(image, 0)

        return faces

    def predict_face_landmark(self, image: np.ndarray, face_bbox: tuple):
        face_shape = self.predictor(image, face_bbox)
        face_points = face_utils.shape_to_np(face_shape)

        return face_points

    def get_eye_landmarks(self, face_landmark: np.ndarray) -> tuple:
        li, le = self.l_eye_idx
        ri, re = self.r_eye_idx

        left_eye_landmark = face_landmark[li:le]
        right_eye_landmark = face_landmark[ri:re]

        return left_eye_landmark, right_eye_landmark

    def drowsiness_ratio(self, left_eye_lm, right_eye_lm):
        left_ratio = self.eye_aspect_ratio(left_eye_lm)
        right_ratio = self.eye_aspect_ratio(right_eye_lm)

        avg_ratio = (left_ratio + right_ratio) / 2.0

        return avg_ratio

    def draw_face_landmarks(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_bbox = self.detect_faces(gray)

        draw = image.copy()
        for bbox in faces_bbox:
            face_landmark = self.predict_face_landmark(image, bbox)

            points = np.expand_dims(face_landmark, axis=1)
            draw = self.draw_contour_points(draw, points)

        return draw

    def draw_eye_landmarks(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_bbox = self.detect_faces(gray)

        draw = image.copy()
        for bbox in faces_bbox:
            face_landmark = self.predict_face_landmark(image, bbox)
            l_eye_lm, r_eye_lm = self.get_eye_landmarks(face_landmark)

            points = np.expand_dims(l_eye_lm, axis=1)
            draw = self.draw_contour_points(draw, points, color=(0, 255, 0))
            points = np.expand_dims(r_eye_lm, axis=1)
            draw = self.draw_contour_points(draw, points, color=(0, 0, 255))

        return draw

    @staticmethod
    def eye_aspect_ratio(eye):
        # Get all vertical landmarks
        left_vertical_lm = dist.euclidean(eye[1], eye[5])
        right_vertical_lm = dist.euclidean(eye[2], eye[4])

        vertical_lm = (left_vertical_lm + right_vertical_lm) / 2.0
        horizontal_lm = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = vertical_lm / horizontal_lm

        return ear

    @staticmethod
    def draw_contour_points(image, points, color: tuple = (0, 255, 0),
                            radius: int = 1):
        draw = image.copy()  # bgr image

        for point in points:
            x, y = point[0]
            draw = cv2.circle(draw, (x, y), color=color, radius=radius,
                              thickness=-1)

        return draw
