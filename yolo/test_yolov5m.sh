python gen-onnx.py
python convert.py
for i in {0..7}
do
    python main.py -id $i 2>&1 | tee yolo.device_$i.log &
done
