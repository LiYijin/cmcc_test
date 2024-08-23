import torch
import os 

if not os.path.exists("./yolov5m.pt"):
    os.system(
        "wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt"
    )
else:
    print("Load Local PTH FILE")
    
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
random_input = torch.randn(24, 3, 640, 640)
# # print(tensor.device.type)
random_input = random_input.to(device)

model.eval()
model.to(device)
torch.onnx.export(model, random_input, "./yolov5m-24-3-640-640.onnx")