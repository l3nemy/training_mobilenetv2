import os
from typing import Any, Callable, Optional
import threading

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torchvision import models
from torchvision.datasets import VisionDataset
from torchvision.transforms import Resize, Normalize, ToTensor, CenterCrop, GaussianBlur
import torchmetrics

import cv2
from PIL import Image

import pytorch_lightning as pl

import onnx2torch


QUANTIZE = True
CAMERA_WIDTH = 224
CAMERA_HEIGHT = 224

if QUANTIZE:
    torch.backends.quantized.engine = 'qnnpack'


classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}


def main():
    ts_model = torch.load("traffic_sign_model.pt")

    #l_model = torch.jit.load("lane_model.pt")
    l_model = onnx2torch.convert('lane.onnx')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 36)
    pp = torchvision.transforms.Compose([
        Resize((32, 32)),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    preprocess = torchvision.transforms.Compose([
        GaussianBlur((3, 3)),
        Resize((200, 66)),
        ToTensor(),
        Normalize([1, 1, 1], [0.5, 0.5, 0.5])
    ])
    with torch.no_grad():
        while True:
            ret, image = cap.read()
            image = image[:, :, [2, 1, 0]]

            pil_image: Image.Image = Image.fromarray(image)
            tm_input_tensor: Tensor = pp(pil_image)
            tm_input_batch = tm_input_tensor.unsqueeze(0)

            l_out: Tensor = l_model(torch.Tensor(img_preprocess(image)))
            angle = l_out[0]

            tm_out: Tensor = ts_model(tm_input_batch)
            # print(np.percentile(out.cpu().numpy()))
            top = list(enumerate(tm_out[0].softmax(dim=0)))
            top.sort(key=lambda x: x[1], reverse=True)
            print(f"angle: {angle}")
            for idx, val in top[:10]:
                print(f"{val.item()*100:.2f}% {classes[idx]}")


def img_preprocess(image):
    height, _, _ = image.shape
    # remove top half of the image, as it is not relavant for lane following
    image = image[int(height/2):, :, :]
    # Nvidia model said it is best to use YUV color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # input image size (200,66) Nvidia model
    image = cv2.resize(image, (200, 66))
    # normalizing, the processed image becomes black for some reason.  do we need this?
    image = image / 255
    return image


if __name__ == '__main__':
    main()
