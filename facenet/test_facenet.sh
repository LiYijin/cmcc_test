#!/bin/bash
for((i=0;i<2;i++)); do
    MUSA_VISIBLE_DEVICES=${i} python3 main_flip_fp16.py -id  ${i} 2>&1 | tee facenet.device_$i.log &
done