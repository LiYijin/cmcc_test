```
conda create --name efficientnet python=3.8
conda activate efficientnet
pip install -r requirements.txt
```

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

You can generate onnx model by:
```
python gen-onnx.py
```
Then, You can test efficientnet accuray and latency only by:
```
for i in {0..7}
do
    MUSA_VISIBLE_DEVICES=$i nohup python main_fp16.py -id $i 2>&1 | tee efficientnet.device_$i.log &
done
```
This script will download model from https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth,
save to fp16 onnx and do inference by batch.
