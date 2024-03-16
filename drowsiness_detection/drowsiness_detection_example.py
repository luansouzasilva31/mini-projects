import cv2
import time
from imutils.video import VideoStream

from drowsiness_detection.drowsiness_detection import DrowsinessDetector


if __name__ == '__main__':
    dat_path = 'drowsiness_detection/shape_predictor_68_face_landmarks.dat'

    video_stream = VideoStream(src=0).start()
    time.sleep(1.0)

    # create objects with initial landmarks
    drowsiness_detector = DrowsinessDetector(dat_path)

    while True:
        frame = video_stream.read()
        faces_landmark = drowsiness_detector.draw_eye_landmarks(frame)
        is_napping = drowsiness_detector.detect_drowsiness(frame)

        # TODO: melhorar detecção de sonolência com análise temporal.

        print(is_napping)

        cv2.imshow("Frame", faces_landmark)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    video_stream.stop()
