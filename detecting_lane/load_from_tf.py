#from .model import LaneModel

import onnx
import onnx2torch

import torch


def main():
    om = onnx.load('lane.onnx')
    torch.onnx.select_model_mode_for_export()
    pm = onnx2torch.convert(om)
    print(pm)


if __name__ == '__main__':
    main()
