from onnx.backend.base import BackendRep
from onnx import TensorProto, shape_inference
import subprocess
import ctypes
import numpy as np
import time
import os
from os.path import dirname, join

from .types import MasterType
from .halide_generator import HalideGraphVisitor


class HalideBackendRep(BackendRep):
    def __init__(self, model, visitor=HalideGraphVisitor()):
        self.name = "{}_{}_{}".format(model.graph.name,
                                      model.model_version,
                                      model.domain.replace('.', '-'))



        for init in model.graph.initializer:
            dims = list(init.dims)
            if init.raw_data:
                onnx_data = np.frombuffer(init.raw_data,
                                          count=np.prod(dims),
                                          dtype=MasterType.from_onnx(init.data_type).np_t)


        value_info = {i.name: i.type for i in list(model.graph.input) +
                      list(model.graph.output) + list(model.graph.value_info)}

        code, objects, headers = visitor.visit(model.graph, value_info)
        print(code)
        print(objects)
        print(headers)
        # for nidx, node in enumerate(model.graph.node):
        #     code, objects, headers = generator.generate(node, value_info)
        #     print(code, objects, headers)
            # print(code, library, header)

    def run(self, inputs, **kwargs):
        print(inputs)
        print(kwargs)
        exit(1)
