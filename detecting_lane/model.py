import glob
from typing import Any, Callable, List, Optional

import numpy as np
from numpy.typing import ArrayLike

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.transforms import Resize, Normalize, ToTensor, CenterCrop
import torchmetrics

import cv2
from PIL import Image

import pytorch_lightning as pl


class LaneModel(pl.LightningModule):
    def __init__(self,  learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        # in_channel = (R, G, B)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2, 2))
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(2, 2))
        self.elu3 = nn.ELU()
        self.conv4 = nn.Conv2d(48, 64, kernel_size=(3, 3))
        self.elu4 = nn.ELU()
        self.dropout1 = nn.Dropout(0.2)

        # FCL
        self.flatten = nn.Flatten()
        self.dropout2 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(64, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 10)
        self.linear3 = nn.Linear(10, 1)

        self.save_hyperparameters(ignore=['model'])

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, actual_labels = batch
        logits = self(features)
        loss = F.mse_loss(logits, actual_labels)
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
