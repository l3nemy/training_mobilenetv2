import os
from typing import Any, Callable, Optional

import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.transforms import Resize, Normalize, ToTensor, CenterCrop
import torchmetrics

import cv2
from PIL import Image

import pytorch_lightning as pl


class TrafficSignModel(pl.LightningModule):
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
