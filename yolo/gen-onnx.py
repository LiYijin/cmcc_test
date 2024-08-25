import torch
import os 

if not os.path.exists("/models/ultralytics_yolov5_master/"):
    print("You should prepare ultralytics yolov5 on /models/, which can be found on /root/.cache/torch/hub/ultralytics_yolov5_master/")
    exit(1)
if not os.path.exists("./yolov5m.pt"):
    os.system(
        "cp /models/yolov5m.pt ."
    )
else:
    print("Load Local PTH FILE")

model = torch.hub.load('/models/ultralytics_yolov5_master//', 'yolov5m', source="local")

device = torch.device("cpu")
# print(device)
random_input = torch.randn(24, 3, 640, 640)
# # print(tensor.device.type)
random_input = random_input.to(device)

model.eval()
model.to(device)
torch.onnx.export(model, random_input, "./yolov5m-24-3-640-640.onnx")