python bert_save_ner.py
python convert.py
for i in {0..7}
do
    MUSA_VISIBLE_DEVICES=$i nohup python main_fp16.py -id $i 2>&1 | tee bert.device_$i.log &
done

