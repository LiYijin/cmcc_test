```bash
git clone https://github.com/espnet/espnet_onnx.git
cd espnet_onnx
git reset --hard 18eb341
cd ..

cd espnet_onnx
patch -p1 < ../export_acc.patch
cp ../multi_batch_beam_search.py espnet_onnx/asr/beam_search
cp ../asr_npu_adapter.py espnet_onnx/asr
cp ../npu_model_adapter.py espnet_onnx/asr
pip install .  #安装espnet_onnx
cd ..

python3 pth2onnx.py #导出模型，正常模型在/root/.cache/espnet_onnx/conformer_test/full 目录下

安装ais-bench/auto-optimizer
参考ais-bench/auto-optimizer安装

python modify_onnx_decoder.py /root/.cache/espnet_onnx/conformer_test/full/xformer_decoder.onnx \
/root/.cache/espnet_onnx/conformer_test/full/xformer_decoder_revise.onnx
python modify_onnx_ctc.py /root/.cache/espnet_onnx/conformer_test/full/ctc.onnx \
/root/.cache/espnet_onnx/conformer_test/full/ctc_dynamic.onnx
python modify_onnx_encoder.py /root/.cache/espnet_onnx/conformer_test/full/xformer_encoder.onnx \
/root/.cache/espnet_onnx/conformer_test/full/xformer_encoder_multibatch.onnx 24

python convert.py #convert ctc lm to fp16, 在文件里指定模型路径

修改${HOME}/.cache/espnet_onnx/conformer_test目录下config配置文件参数，
以及修改对应的weight参数中的ctc, decoder, lm文件路径。设置beam size的大小为2;

./run.sh
```