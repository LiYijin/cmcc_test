from models.utils import tensorflow2pytorch
import os

if not os.path.exists("/models/resnet50_b16x8_cifar100_20210528-67b58a1b.pth"):
    os.system(
        "wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth"
    )
else:
    print("Load Local PTH FILE")

tensorflow2pytorch.tensorflow2pytorch()