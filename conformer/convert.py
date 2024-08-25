#Convert to ONNX ModelProto object and save model binary file:
from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
import onnx
new_onnx_model = convert_float_to_float16_model_path('/root/.cache/espnet_onnx/conformer_test/full/ctc_dynamic.onnx')
onnx.save(new_onnx_model, '/root/.cache/espnet_onnx/conformer_test/full/ctc_24_fp16.onnx')

new_onnx_model = convert_float_to_float16_model_path('/root/.cache/espnet_onnx/conformer_test/full/transformer_lm.onnx')
onnx.save(new_onnx_model, '/root/.cache/espnet_onnx/conformer_test/full/transformer_lm_fp16.onnx')
    
