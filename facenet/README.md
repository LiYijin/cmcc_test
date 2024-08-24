## README

```shell
cd gen-onnx
conda create --name facenet-convert python=3.6 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
pip install -r requirement.txt
python3 convert.py
python3 gen-onnx.py # gen onnx file in ../model/facenet-fp32.onnx
conda deactivate
cd ..
conda create --name facenet python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
pip install -r requirement.txt
bash ./test_facenet.sh
```