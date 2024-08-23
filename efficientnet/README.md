#Prepare dataset.
##download:
Dataset used is imagenet1k 2012, you can download it from https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar,
This download requires your request by login.
##Preprocess:
You can preprocess this dataset by
```
bash extra_dataset.sh
```

#Test efficientnet.

You can test efficientnet accuray and latency only by:
```
bash test_efficientnet.sh
```
This script will download model from https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth,
save to fp16 onnx and do inference by batch.

