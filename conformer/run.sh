MUSA_VISIBLE_DEVICES=0 python om_val.py --dataset_path ${dataset_path}/wav/test --model_path ${HOME}/.cache/espnet_onnx/conformer_test --batch_encoder 24 --batch_decoder 24  --num_process_encoder 1 --num_process_decoder 1 --d_id 0 &
MUSA_VISIBLE_DEVICES=1 python om_val.py --dataset_path ${dataset_path}/wav/test --model_path ${HOME}/.cache/espnet_onnx/conformer_test --batch_encoder 24 --batch_decoder 24  --num_process_encoder 1 --num_process_decoder 1 --d_id 1 &
MUSA_VISIBLE_DEVICES=2 python om_val.py --dataset_path ${dataset_path}/wav/test --model_path ${HOME}/.cache/espnet_onnx/conformer_test --batch_encoder 24 --batch_decoder 24  --num_process_encoder 1 --num_process_decoder 1 --d_id 2 &
MUSA_VISIBLE_DEVICES=3 python om_val.py --dataset_path ${dataset_path}/wav/test --model_path ${HOME}/.cache/espnet_onnx/conformer_test --batch_encoder 24 --batch_decoder 24  --num_process_encoder 1 --num_process_decoder 1 --d_id 3 &
MUSA_VISIBLE_DEVICES=4 python om_val.py --dataset_path ${dataset_path}/wav/test --model_path ${HOME}/.cache/espnet_onnx/conformer_test --batch_encoder 24 --batch_decoder 24  --num_process_encoder 1 --num_process_decoder 1 --d_id 4 &
MUSA_VISIBLE_DEVICES=5 python om_val.py --dataset_path ${dataset_path}/wav/test --model_path ${HOME}/.cache/espnet_onnx/conformer_test --batch_encoder 24 --batch_decoder 24  --num_process_encoder 1 --num_process_decoder 1 --d_id 5 &
MUSA_VISIBLE_DEVICES=6 python om_val.py --dataset_path ${dataset_path}/wav/test --model_path ${HOME}/.cache/espnet_onnx/conformer_test --batch_encoder 24 --batch_decoder 24  --num_process_encoder 1 --num_process_decoder 1 --d_id 6 &
MUSA_VISIBLE_DEVICES=7 python om_val.py --dataset_path ${dataset_path}/wav/test --model_path ${HOME}/.cache/espnet_onnx/conformer_test --batch_encoder 24 --batch_decoder 24  --num_process_encoder 1 --num_process_decoder 1 --d_id 7 &
