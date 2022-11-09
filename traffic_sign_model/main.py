from .model import DataModule, TrafficSignModel

import os
from time import time
from typing import Any, Callable, Optional

import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import models
from torchvision.datasets import VisionDataset
from torchvision.transforms import Resize, Normalize, ToTensor, CenterCrop

import cv2
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, QuantizationAwareTraining
from pytorch_lightning.loggers import CSVLogger

import torch

torch.backends.quantized.engine = 'qnnpack'


BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
NUM_WORKERS = 4
FEATURES = 43

"""
class DataSet(Dataset):
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
        img = cv2.resize(img_file, self.img_size, self.img_size)
        img = img.astype(np.float64)
        label = self.img_labels[idx]

        return {
            'x': torch.tensor(img, dtype=torch.float),
            'y': torch.tensor(label, dtype=torch.long),
        }
 """


class TrafficSignDataset(VisionDataset):
    """meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"""

    def __init__(self, root='./', train=True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        self.train = train
        if train:
            folder_name = "Train"
        else:
            folder_name = "Test"

        self.csv_data = pd.read_csv(
            os.path.join(self.root, folder_name + '.csv'))
        """
        Width
        Height
        Roi.X1
        Roi.Y1
        Roi.X2
        Roi.Y2
        ClassId
        Path
        """

    def __len__(self) -> int:
        return self.csv_data.shape[0]

    def __getitem__(self, index: int) -> Any:
        d = self.csv_data.iloc[index]
        data = {
            "width": d["Width"],
            "height": d["Height"],
            "x1":  d["Roi.X1"],
            "y1": d["Roi.Y1"],
            "x2": d["Roi.X2"],
            "y2": d["Roi.Y2"],
            "cid": d["ClassId"],
            "path": d["Path"],
        }
        img = cv2.imread(os.path.join(self.root, data['path']))
        img = img[data['y1']: data['y2'],
                  data['x1']: data['x2'], :]  # crop image
        img = img[:, :, ::-1]  # convert bgr into rgb
        img = Image.fromarray(img)

        target = data['cid']

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [
            torch.clone(img).detach(),
            torch.tensor(target, dtype=torch.long),
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
        train = TrafficSignDataset(root=self.path,
                                   train=True,
                                   transform=self.train_transform)
        self.test = TrafficSignDataset(root=self.path,
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
    logger = CSVLogger("logs/", name="stop_model")

    datamodule = DataModule('./datasets')

    orig_model = models.quantization.mobilenet_v2(
        pretrained=False, quantize=QUANTIZE)
    orig_model.classifier[-1] = torch.nn.Linear(
        in_features=1280,
        out_features=FEATURES
    )
    model = TrafficSignModel(orig_model, LEARNING_RATE)

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[
            EarlyStopping('vaild_acc', mode='max'),
            ModelCheckpoint("./checkpoints"),
            QuantizationAwareTraining()
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
    torch.jit.save(script, "traffic_sign_model.pt")


if __name__ == '__main__':
    main()
