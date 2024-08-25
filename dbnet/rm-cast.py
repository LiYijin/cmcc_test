#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx_graphsurgeon as gs
import onnx
import numpy as np

input = gs.Variable(name="input", dtype=np.float16, shape=(24, 3, 736, 1280))

graph = gs.import_onnx(onnx.load("./model/model-fp16-static-swap.onnx"))

cast_node = [node for node in graph.nodes if node.name == "graph_input_cast0"][0]
conv_node = [node for node in graph.nodes if node.name == "p2o.Conv.0"][0]
inp_node = cast_node.inputs[0]


conv_node.inputs[0] = input
# Reconnect the input node to the output tensors of the fake node, so that the first identity
# node in the example graph now skips over the fake node.
print(inp_node)
inp_node.outputs = cast_node.outputs
cast_node.outputs.clear()

# Remove the fake node from the graph completely

graph.inputs = [input]
graph.cleanup().toposort()




# 
sigmoid = [node for node in graph.nodes if node.name == "p2o.Sigmoid.0"][0]
graph.outputs = [sigmoid.outputs[0]]

graph.cleanup().toposort()





model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
onnx.save(model, "./model/dbnet-fp16.onnx")
