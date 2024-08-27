
#!/bin/bash
python3 tools/export_model.py -c ./configs/det/det_mv3_db.yml -o Global.pretrained_model=./det_mv3_db_v2.0_train/best_accuracy Global.save_inference_dir=./infer-mv3-db

paddle2onnx --model_dir ./infer-mv3-db/ \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ./infer-mv3-db/model-fp16.onnx \
    --opset_version 11 \
    --enable_onnx_checker True \
    --export_fp16_model True


