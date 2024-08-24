#Convert to ONNX ModelProto object and save model binary file:
from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
import onnx
new_onnx_model = convert_float_to_float16_model_path('ctc_24.onnx')
onnx.save(new_onnx_model, 'ctc_24_fp16.onnx')

new_onnx_model = convert_float_to_float16_model_path('transformer_lm.onnx')
onnx.save(new_onnx_model, 'transformer_lm_fp16.onnx')
    