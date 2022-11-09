import cv2
import torch
import torchvision
import numpy as np
from torchvision import transforms

import torchmetrics

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while cv2.waitKey(33) < 0:
        frames = []
        ret, frame = cap.read()
        # frames.append(frame)

        #median_frame = np.median(frames, axis=0)
        cv2.imshow("VideoFrame", frame)

    cap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
