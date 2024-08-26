1. Origin bert model can be dowloaded on:
https://huggingface.co/google-bert/bert-base-chinese
by the instructions on this website:
git lfs install
git clone https://huggingface.co/google-bert/bert-base-chinese

You shoul put original bert model on /models/bert-base-chinese, which contains config/tokenizer, etc.

Pretrained model shoule be put on /models/epoch_3_valid_macrof1_95.812_microf1_95.904_weights.bin 

2. Then,
You can generate onnx model and convert to fp16 model by:
```
python gen-onnx.py
python convert.py
```

3. Then, You can test efficientnet accuray and latency only by:
```
for i in {0..7}
do
    MUSA_VISIBLE_DEVICES=$i nohup python main_fp16.py -id $i 2>&1 | tee efficientnet.device_$i.log &
done
```
