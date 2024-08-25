from models.inception_resnet_v1 import InceptionResnetV1
import torch

# Load pretrained resnet model
resnet = InceptionResnetV1(
    classify=False,
    pretrained='casia-webface'
)

random_input = torch.randn(64, 3, 160, 160)
torch.onnx.export(resnet, random_input, "../model/facenet-fp32.onnx")

