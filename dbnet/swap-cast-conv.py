#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx_graphsurgeon as gs
import numpy as np
import onnx

graph = gs.import_onnx(onnx.load("./model/model-fp16-static.onnx"))

# 1. Remove the `b` input of the add node
concat = [node for node in graph.nodes if node.name == "p2o.Concat.0"][0]
old_conv = [node for node in graph.nodes if node.name == "p2o.Conv.71"][0]

print(concat)
print(old_conv)
# 3. Add an identity after the add node
cast_out = gs.Variable("sigma_cast_out", dtype=np.float32)
cast = gs.Node(op="Cast", inputs=concat.outputs, outputs=[cast_out], name="sigma-cast")
cast.attrs['to'] = cast_out.dtype


# X = gs.Variable(name="X", dtype=np.float32, shape=(1, 3, 224, 224))
# Since W is a Constant, it will automatically be exported as an initializer
w_np = np.fromfile('w-conv.txt', dtype=np.float16).reshape(64, 256, 3, 3).astype(np.float32)
W = gs.Constant(name="W", values=w_np)

conv_out = gs.Variable(name="Y", dtype=np.float32, shape=(24, 64, 184, 320))

conv = gs.Node(op="Conv", inputs=[cast_out, W], outputs=[conv_out], name="sigma-conv")

# conv.attrs["auto_pad"] = old_conv.attrs["auto_pad"]
conv.attrs["dilations"] = old_conv.attrs["dilations"]
conv.attrs["group"] = old_conv.attrs["group"]
conv.attrs["kernel_shape"] = old_conv.attrs["kernel_shape"]
conv.attrs["pads"] = old_conv.attrs["pads"]
conv.attrs["strides"] = old_conv.attrs["strides"]




graph.nodes.append(cast)
graph.nodes.append(conv)


bn_node = [node for node in graph.nodes if node.name == "p2o.BatchNormalization.47"][0]
bn_node.inputs[0] = conv_out

# 5. Remove unused nodes/tensors, and topologically sort the graph
# ONNX requires nodes to be topologically sorted to be considered valid.
# Therefore, you should only need to sort the graph when you have added new nodes out-of-order.
# In this case, the identity node is already in the correct spot (it is the last node,
# and was appended to the end of the list), but to be on the safer side, we can sort anyway.
graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "./model/model-fp16-static-swap.onnx")
