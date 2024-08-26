from onnxruntime.mix_precision_opt import run

run("./gen-onnx/infer-mv3-db/model-fp16.onnx", "model/dbnet-fp16.onnx")