#Convert to ONNX ModelProto object and save model binary file:
from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
import onnx
new_onnx_model = convert_float_to_float16_model_path('bert_ner_fp32_64-opset17.onnx')
onnx.save(new_onnx_model, 'bert_ner_fp16_64.onnx')