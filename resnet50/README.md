```
conda create --name resnet50 python=3.8
conda activate resnet50
pip install -r requirements.txt
```
You can generate onnx model by:
```shell
python gen-onnx.py -P fp16
```
And then, test inference by:
```shell
for i in {0..7}
do
    MUSA_VISIBLE_DEVICES=$i nohup python main.py -P fp16 -id $i 2>&1 | tee resnet.device_$i.log &
done
```
