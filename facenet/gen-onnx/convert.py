from models.utils import tensorflow2pytorch
import os
if not os.path.exists("./data/20180408-102900/"):
    os.system(
        "cp -r /models/20180408-102900/  ./data/."
    )

tensorflow2pytorch.tensorflow2pytorch()
