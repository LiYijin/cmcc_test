#!/bin/bash
for((i=0;i<2;i++)); do
    MUSA_VISIBLE_DEVICES=${i} python3 eval_fp16.py -c det_mv3_db.yml -id ${i} 2>&1 | tee dbnet.device_$i.log &
done