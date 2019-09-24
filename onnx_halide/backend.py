import numpy as np
import onnx
import onnx.utils
from onnx.backend.base import Backend
from onnx import helper, TensorProto, shape_inference

import subprocess
from .backend_rep import HalideBackendRep

class HalideBackend(Backend):

    # @classmethod
    # def run_node(cls, node, input, device='CPU'):
    #     input_tensors = []

    #     if len(node.input) != len(input):
    #         raise ValueError(
    #             "Unexpected Input Size: Op_Type = {0}, Expected = {1}, Received = {2}"
    #             .format(node.op_type, len(node.input), len(input))
    #             )

    #     for i in range(len(input)):
    #         input_tensors.append(helper.make_tensor_value_info(node.input[i], TensorProto.FLOAT, input[i].shape))

    #     onnx_graph = helper.make_graph([node], "test_{}".format(node.op_type), input_tensors, [])
    #     onnx_model = helper.make_model(onnx_graph)
    #     return HalideBackend.run_model(onnx_model, input, device)

    @classmethod
    def run_model(cls, model, input, device='CPU'):
        rep = HalideBackend.prepare(model, device)
        exit(1)
        return rep.run(input)

    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        return HalideBackendRep(onnx.utils.polish_model(model))

    @classmethod
    def supports_device(cls, device='CPU'):
        return device in ['CPU']


run_model = HalideBackend.run_model
run_node = HalideBackend.run_node
supports_device = HalideBackend.supports_device
prepare = HalideBackend.prepare
