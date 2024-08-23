import torch
import os 
import time
import timm

if not os.path.exists("./efficientnetv2_t_agc-3620981a.pth"):
    os.system(
        "wget https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth"
    )
else:
    print("Load Local PTH FILE")

efficientnet=timm.create_model('efficientnetv2_rw_t', pretrained=False, num_classes=1000)
efficientnet.load_state_dict(torch.load("./efficientnetv2_t_agc-3620981a.pth"))
efficientnet = efficientnet.half()
efficientnet.eval()

device = torch.device("cpu")

model = efficientnet

batch = 24

random_input = torch.randn(batch, 3, 288, 288)
random_input = random_input.half()

name = f"efficientnetv2_rw_t_fp16_" + str(batch) + ".onnx"
torch.onnx.export(model,random_input,name, opset_version=11)    
