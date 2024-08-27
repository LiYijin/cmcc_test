```bash
#在cmcc_test的目录下进行操作
git clone https://github.com/hochen1/espnet_onnx.git
cd espnet_onnx
python install .

# 安装auto-optimizer
# 参考https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer
git clone https://gitee.com/ascend/msadvisor.git
cd msadvisor/auto-optimizer
python3 -m pip install .

pip install typeguard==2.13.3
pip install typeguard==2.13.3
pip install espnet==202402
pip install espnet-model-zoo==0.1.7
pip install torchaudio
pip install onnxmltools==1.12.0
pip install psutil
#所以缺少的python依赖按照requirements_new.txt版本安装

python download.py

#导出模型,文件中要更改conformer模型.zip文件的路径，正常模型在/root/.cache/espnet_onnx/conformer_test/full 目录下
#注意check pth2onnx.py中的所有路径
python3 pth2onnx.py 

# 修改${HOME}/.cache/espnet_onnx/conformer_test目录下config.yaml配置文件参数，以及修改对应参数中的encoder, ctc, decoder, lm文件路径。

# 修改config.yaml的beam size 为 2

#最后要在om_val.py中检查373行的text_value变量，要更改为'/data/data_aishell/transcript/aishell_transcript_v0.8.txt'

#在run.sh脚本中，需要修改数据集路径
./run.sh
```
