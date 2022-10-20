import os
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.datasets import VisionDataset
from torchvision.transforms import Resize, Normalize, ToTensor, CenterCrop
import torchmetrics

import cv2
from PIL import Image

import pytorch_lightning as pl

QUANTIZE = True

if QUANTIZE:
    torch.backends.quantized.engine = 'qnnpack'


class Model(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        self.save_hyperparameters(ignore=['model'])

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, actual_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, actual_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, actual_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, actual_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)

        self.model.eval()
        with torch.no_grad():
            _, actual_labels, predicted_labels = self._shared_step(batch)
        self.train_acc.update(predicted_labels, actual_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)

        self.model.train()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, actual_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.valid_acc(predicted_labels, actual_labels)
        self.log("valid_acc", self.valid_acc, on_epoch=True,
                 on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, actual_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, actual_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


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
    orig_model = models.quantization.mobilenet_v2(
        pretrained=False, quantize=False,)

    model: Model = Model(orig_model, 0.01)
    orig_model.classifier[-1] = torch.nn.Linear(
        in_features=1280,
        out_features=43
    )
    model.load_state_dict(torch.load('./model'))
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",
    )
    model.eval()
    # model = torch.jit.script(model)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv2.CAP_PROP_FPS, 36)
    pp = torchvision.transforms.Compose([
        Resize((32, 32)),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    with torch.no_grad():
        while True:
            ret, image = cap.read()
            image = image[:, :, [2, 1, 0]]
            permuted = image

            image = Image.fromarray(image)
            input_tensor = pp(image)
            input_batch = input_tensor.unsqueeze(0)

            out: torch.Tensor = model(input_batch)
            # print(np.percentile(out.cpu().numpy()))
            top = list(enumerate(out[0].softmax(dim=0)))
            top.sort(key=lambda x: x[1], reverse=True)
            for idx, val in top[:10]:
                print(f"{val.item()*100:.2f}% {classes[idx]}")


if __name__ == '__main__':
    main()
