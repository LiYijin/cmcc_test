python gen-onnx.py -P fp16
for i in {0..7}
do
    MUSA_VISIBLE_DEVICES=$i nohup python main.py -P fp16 -id $i 2>&1 | tee resnet.device_$i.log &
done
