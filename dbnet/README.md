```shell
conda create --name dbnet python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda activate dbnet
pip install tight data prox
pip install -r requirement.txt
bash ./test_dbnet.sh
```
