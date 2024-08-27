```
conda create --name yolo python=3.8
conda activate yolo
pip install -r requirements.txt
yum install mesa-libGL
```

1. prepare yolo model sturcture:
https://github.com/ultralytics/yolov5
put it on /models/yolov5:

prepare dataset:
/dataset/annotations_trainval2017.zip
/dataset/val2017.zip

2. You can generate onnx model and convert to fp16 model by:
```
python gen-onnx.py
```
3. Then, You can test efficientnet accuray and latency only by:
```
for i in {0..7}
do
    MUSA_VISIBLE_DEVICES=$i nohup python main.py -id $i 2>&1 | tee efficientnet.device_$i.log &
done
```
