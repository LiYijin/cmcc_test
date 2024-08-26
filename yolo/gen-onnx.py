import torch
import os 
from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
import onnx

if not os.path.exists("./yolov5m.pt"):
    os.system(
        "cp /models/yolov5m.pt ."
    )
else:
    print("Load Local PTH FILE")

# This is for model download online
# model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
# This is for model offline
model = torch.hub.load('/models/yolov5/', 'yolov5m', source="local")

device = torch.device("cpu")
# print(device)
random_input = torch.randn(24, 3, 640, 640)
# # print(tensor.device.type)
random_input = random_input.to(device)

model.eval()
model.to(device)
torch.onnx.export(model, random_input, "./yolov5m-24-3-640-640.onnx")
# convert to fp16
new_onnx_model = convert_float_to_float16_model_path('yolov5m-24-3-640-640.onnx')
onnx.save(new_onnx_model, 'yolov5m-24-3-640-640-fp16.onnx')