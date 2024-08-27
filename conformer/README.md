```bash
# 起conda环境
conda create -n conformer python=3.8
conda activate conformer

# 在cmcc_test的目录下进行操作,安装espnet_onnx
git clone https://github.com/hochen1/espnet_onnx.git
cd espnet_onnx
pip install .

# 安装auto-optimizer
# 参考https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer
git clone https://gitee.com/ascend/msadvisor.git
cd msadvisor/auto-optimizer
python3 -m pip install .

#如果有缺少的依赖按照requirements_new.txt版本安装
pip install typeguard==2.13.3 espnet==202402 espnet-model-zoo==0.1.7 torchaudio onnxmltools==1.12.0 psutil

python download.py

#导出模型,文件中要更改conformer模型.zip文件的路径，正常模型在/root/.cache/espnet_onnx/conformer_test/full 目录下
#注意check pth2onnx.py中的所有路径
python3 pth2onnx.py 

# 修改${HOME}/.cache/espnet_onnx/conformer_test目录下文件如下：
#   2   beam_size: 2
#   8   model_path: /root/.cache/espnet_onnx/conformer_test/full/ctc_24_fp16.onnx
#  11   model_path: /root/.cache/espnet_onnx/conformer_test/full/xformer_decoder_revise.onnx
#  37   model_path: /root/.cache/espnet_onnx/conformer_test/full/xformer_encoder_multibatch.onnx
#  46   model_path: /root/.cache/espnet_onnx/conformer_test/full/transformer_lm_fp16.onnx

#最后要在om_val.py中检查373行的text_value变量，要更改为'/data/data_aishell/transcript/aishell_transcript_v0.8.txt'

#在run.sh脚本中，需要修改数据集路径
dataset_path=/dataset/data_aishell/ ./run.sh
```
