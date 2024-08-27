## 测试
创建conda环境
```
conda create --name efficientnet python=3.8
conda activate efficientnet
pip install -r requirements.txt
```

转成onnx模型
```shell
python gen-onnx.py
```
测试推理
```shell
bash test_efficientnet.sh
```

退出conda环境
```
conda deactivate
```

## Tips
* 数据集imagenet1k 2012下载：https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar，需要申请下载
* 数据集解压提取可以通过
```
bash extra_dataset.sh
```
