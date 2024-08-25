```bash
#在cmcc_test的目录下进行操作
git clone https://github.com/espnet/espnet_onnx.git
cd espnet_onnx
git reset --hard 18eb341

git apply -p1 < ../export_acc.patch
cp ../multi_batch_beam_search.py espnet_onnx/asr/beam_search
cp ../asr_npu_adapter.py espnet_onnx/asr
cp ../npu_model_adapter.py espnet_onnx/asr
pip install .  #安装espnet_onnx，这里会安装一些python包
cd ..

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

#导出模型,文件中要更改conformer模型.zip文件的路径，正常模型在/root/.cache/espnet_onnx/conformer_test/full 目录下
python3 pth2onnx.py 

# 脚本后面指定原模型路径和生成模型文件路径
python modify_onnx_decoder.py /root/.cache/espnet_onnx/conformer_test/full/xformer_decoder.onnx \
/root/.cache/espnet_onnx/conformer_test/full/xformer_decoder_revise.onnx

python modify_onnx_ctc.py /root/.cache/espnet_onnx/conformer_test/full/ctc.onnx \
/root/.cache/espnet_onnx/conformer_test/full/ctc_dynamic.onnx

#这里要指定batch size 为 24
python modify_onnx_encoder.py /root/.cache/espnet_onnx/conformer_test/full/xformer_encoder.onnx \
/root/.cache/espnet_onnx/conformer_test/full/xformer_encoder_multibatch.onnx 24

#convert ctc lm to fp16，在文件中更改以前的之前的文件路径，和更新后的文件路径
python convert.py

# 修改${HOME}/.cache/espnet_onnx/conformer_test目录下config.yaml配置文件参数，以及修改对应参数中的encoder, ctc, decoder, lm文件路径。

# 修改config.yaml的beam size 为 2

#最后要在om_val.py中检查373行的text_value变量，要更改为'/data/data_aishell/transcript/aishell_transcript_v0.8.txt'

#在run.sh脚本中，需要修改数据集路径
./run.sh
```


