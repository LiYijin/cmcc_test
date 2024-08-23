python gen-onnx.py -P fp16
for i in {0..7}
do
    python main.py -P fp16 -id $i &
done