python gen-onnx.py
for i in {0..1}
do
    MUSA_VISIBLE_DEVICES=$i nohup python main_fp16.py -id $i 2>&1 | tee efficientnet.device_$i.log &
done
