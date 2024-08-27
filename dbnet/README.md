```shell
conda create --name dbnet python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda activate dbnet
pip install common tight data prox onnx_graphsurgeon dual
pip install -r requirement.txt
cd gen-onnx
pip install -r requirement.txt
pip install paddle2onnx
bash convert.sh
cd ..
mkdir -p model && python3 mix-precision-opt.py
mkdir -p /root/dbnet-infer && cp -r /dataset/icdar2015/ /root/dbnet-infer/
bash ./test_dbnet.sh
```
