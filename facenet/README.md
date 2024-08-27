## README

1. 使用mtcnn预处理lfw数据集

```shell
cd facenet-tf
conda create -n facenet-mtcnn python=3.5
conda activate facenet-mtcnn
pip install --upgrade pip
pip install -r requirements.txt
pip install opencv-python-headless numpy==1.16.2 scipy==1.2.1
export PYTHONPATH=./src
bash preprocess_lfw.sh
conda deactivate
cd ../
```

2. 将tf模型转为pytorch模型

```shell
cd gen-onnx
conda create --name facenet-convert python=3.6 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda activate facenet-convert
pip install --trusted-host pypi.python.org --upgrade pip
pip install -r requirements.txt
python3 convert.py
conda deactivate
cd ..
```

3. 将pytorch模型转为onnx模型，并测试
```shell
conda create --name facenet python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda activate facenet
pip install -r requirements.txt
cd gen-onnx
python3 gen-onnx.py
cd ..
bash ./test_facenet.sh
conda deactivate
```


### Tips:

* facenet-tf 下载自 https://github.com/davidsandberg/facenet 我们修复了requirements.txt，并增加了preprocess_lfw.sh脚本
* tf pb文件转pytoch pt文件 参考https://github.com/timesler/facenet-pytorch
* load pt文件生成onnx模型的模型backbone 参考https://github.com/timesler/facenet-pytorch
