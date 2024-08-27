from models.inception_resnet_v1 import InceptionResnetV1
import torch
import os

# Load pretrained resnet model
resnet = InceptionResnetV1(
    classify=False,
    pretrained='casia-webface'
)
os.system("mkdir -p ../model && mkdir -p ../data")
os.system("cp -r /dataset/lfw_mtcnnpy_160 ../data/")
os.system("cp  /dataset/pairs.txt ../data/")
random_input = torch.randn(64, 3, 160, 160)
torch.onnx.export(resnet, random_input, "../model/facenet-fp32.onnx")

import onnx
from onnxconverter_common import float16

model = onnx.load("../model/facenet-fp32.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "../model/facenet-fp16.onnx")

