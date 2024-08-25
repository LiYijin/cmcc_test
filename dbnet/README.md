```shell
conda create --name dbnet python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda activate dbnet
cd gen-onnx
bash convert.sh
cd ..
cp gen-onnx/infer-mv3-db/model-fp16.onnx model/model-fp16-base.onnx
python3 mix-precision-opt.py
pip install -r requirement.txt
bash ./test_dbnet.sh
```
