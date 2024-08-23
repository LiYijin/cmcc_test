for i in {0..7}
do
    python main_fp16.py -id $i 2>&1 | tee bert.device_$i.log &
done

