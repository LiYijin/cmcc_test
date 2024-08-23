#Convert to ONNX ModelProto object and save model binary file:
from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
import onnx
new_onnx_model = convert_float_to_float16_model_path('yolov5m-1-3-640-640.onnx')
onnx.save(new_onnx_model, 'yolov5m-1-3-640-640-fp16.onnx')