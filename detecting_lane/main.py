from .model import LaneModel, DataModule

from time import time
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

import cv2
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, QuantizationAwareTraining
from pytorch_lightning.loggers import CSVLogger

torch.backends.quantized.engine = 'qnnpack'

BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
NUM_WORKERS = 4


class LaneDataset(VisionDataset):
    def __init__(self, path: str, img_size: int, labels: List[str]) -> None:
        self.path = path
        self.img_size = img_size
        self.labels = labels

        img_names = []
        img_labels = []

        for i, label in enumerate(labels):
            imgs = glob.glob(
                '/'.join([glob.escape(path), label])
            )
            [img_labels.append(i) for _ in imgs]
            img_names += imgs

        self.img_names = img_names
        self.img_labels = img_labels

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx) -> ArrayLike:
        img_file = cv2.imread(self.path + self.img_names[idx])
        img = cv2.cvtColor(img_file, cv2.COLOR_BGR2HSV)
        img = cv2.resize(img, self.img_size, self.img_size)
        img = img.astype(np.float64)
        label = self.img_labels[idx]

        return [
            torch.clone(img),
            torch.tensor(label, dtype=torch.long),
        ]


class DataModule(pl.LightningDataModule):
    def __init__(self, path='./'):
        super().__init__()
        self.path = path

    def prepare_data(self) -> None:
        #TrafficSignDataset(root=self.path, download=True)
        self.train_transform = torchvision.transforms.Compose([
            Resize((32, 32)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.test_transform = torchvision.transforms.Compose([
            Resize((32, 32)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        train = LaneDataset(root=self.path,
                            train=True,
                            transform=self.train_transform)
        self.test = LaneDataset(root=self.path,
                                train=False,
                                transform=self.test_transform)

        # total datas count: 39209
        self.train, self.valid = random_split(train, lengths=[34209, 5000])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=BATCH_SIZE,
            drop_last=True,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )


def main():
    logger = CSVLogger("logs/", name="lane_model")

    datamodule = DataModule('./datasets')

    model = LaneModel(learning_rate=LEARNING_RATE)

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[
            EarlyStopping('vaild_acc', mode='max'),
            ModelCheckpoint("./checkpoints"),
            QuantizationAwareTraining(),
        ],
        accelerator="auto",
        devices="auto",
        logger=logger,
        log_every_n_steps=100
    )

    start = time()
    trainer.fit(model, datamodule=datamodule)

    finish = (time() - start) / 60
    print(f"Took {finish:.3f} min")

    quantized_model = torch.quantization.convert(model)
    script = quantized_model.to_torchscript()
    torch.jit.save(script, 'model.pt')


if __name__ == '__main__':
    main()
