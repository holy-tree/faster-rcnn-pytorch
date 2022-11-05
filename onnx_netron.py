import torchvision.models as models
import torch

import onnx
import onnx.utils
import onnx.version_converter


data = torch.randn(2, 3, 256, 256)
net = models.resnet34()

# torch.save()

torch.onnx.export(
    net,
    data,
    './model.onnx',
    export_params=True,
    opset_version=8,
)

