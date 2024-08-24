import onnx
from onnxconverter_common import float16

model = onnx.load("./model/facenet-fp32.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "./model/facenet-fp16.onnx")
