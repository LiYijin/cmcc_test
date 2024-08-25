## README
First, you should use mtcnn to proprocess flw raw jpg.
facenet-tf is cloned from https://github.com/davidsandberg/facenet.
We fix requirements.txt and add preprocess_lfw.sh script
```shell
cd facenet-tf
conda create -n facenet-mtcnn python=3.5
pip install -r requirements.txt
bash preprocess_lfw.sh
```

Then, generate onnx model and test.
```shell
cd gen-onnx
conda create --name facenet-convert python=3.6 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
pip install -r requirement.txt
python3 convert.py
python3 gen-onnx.py # gen onnx file in ../model/facenet-fp32.onnx
conda deactivate
cd ..
python3 convert-fp16.py
conda create --name facenet python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
pip install -r requirement.txt
bash ./test_facenet.sh
```
