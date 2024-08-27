创建conda环境
```
conda create --name resnet50 python=3.8
conda activate resnet50
pip install -r requirements.txt
```

转成onnx模型
```shell
python gen-onnx.py -P fp16
```
测试推理
```shell
bash test_resnet50.sh
```

退出conda环境
```
conda deactivate
```