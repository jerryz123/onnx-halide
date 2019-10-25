import numpy as np
import onnx
import onnx.utils
from onnx.backend.base import Backend
from onnx import helper, TensorProto, shape_inference
import re

import subprocess
from .backend_rep import HalideBackendRep

from onnx.onnx_ml_pb2 import ModelProto
from onnx_halide.backend_rep import HalideBackendRep
from google.protobuf.pyext._message import RepeatedScalarContainer

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
    def run_model(cls, model: ModelProto, input, device='CPU'):
        rep = HalideBackend.prepare(model, device)
        exit(1)
        return rep.run(input)

    @classmethod
    def prepare(cls, model: ModelProto, device: str = 'CPU', **kwargs) -> HalideBackendRep:
        return HalideBackendRep(onnx.utils.polish_model(cls.sanitize_model(model)))

    @classmethod
    def sanitize_string(cls, str) -> str:
        # For production ready (tm) code this probably needs to be more robust
        # Maybe just use escaped utf encoding or hash it
        # Must be an idempotent operation
        scrubbed = re.sub(r"[!@#/$%^&*()\-+\[\]]", "_", str)
        if scrubbed[:2] != "v_":
            scrubbed = "v_" + scrubbed
        return scrubbed

    @classmethod
    def sanitize_model(cls, model: ModelProto) -> ModelProto:
        '''Mutates model to sanitize layer names'''
        for i in range(len(model.graph.node)):
            for j in range(len(model.graph.node[i].input)):
                model.graph.node[i].input[j] = cls.sanitize_string(model.graph.node[i].input[j])
            for j in range(len(model.graph.node[i].output)):
                model.graph.node[i].output[j] = cls.sanitize_string(model.graph.node[i].output[j])

        for i in range(len(model.graph.input)):
            model.graph.input[i].name = cls.sanitize_string(model.graph.input[i].name)
        for i in range(len(model.graph.output)):
            model.graph.output[i].name = cls.sanitize_string(model.graph.output[i].name)
        for i in range(len(model.graph.initializer)):
            model.graph.initializer[i].name = cls.sanitize_string(model.graph.initializer[i].name)
        return model


    @classmethod
    def supports_device(cls, device: str = 'CPU') -> bool:
        return device in ['CPU']


run_model = HalideBackend.run_model
run_node = HalideBackend.run_node
supports_device = HalideBackend.supports_device
prepare = HalideBackend.prepare
