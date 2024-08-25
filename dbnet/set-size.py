import onnx
from onnx.tools import update_model_dims

model = onnx.load("./model/model-fp16-base.onnx")
# Here both "seq", "batch" and -1 are dynamic using dim_param.
variable_length_model = update_model_dims.update_inputs_outputs_dims(model, {"x": [24, 3, 736, 1280]}, {"save_infer_model/scale_0.tmp_0": [24, 1, 736, 1280]})

onnx.save(variable_length_model, "./model/model-fp16-static.onnx")

