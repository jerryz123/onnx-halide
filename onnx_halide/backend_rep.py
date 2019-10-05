from onnx.backend.base import BackendRep
from onnx import TensorProto, shape_inference
import subprocess
import ctypes
import numpy as np
import time
import os
from os.path import dirname, join, abspath

from .types import VI, from_onnx_t
from .halide_generator import HalideGraphVisitor
from .environment_link import Environment

from numpy import ndarray
from onnx.onnx_ml_pb2 import ModelProto
from typing import List, Type

class HalideBackendRep(BackendRep):
    def __init__(self, model: ModelProto, temp_dir: str = "build", visitor: Type[HalideGraphVisitor] = HalideGraphVisitor) -> None:
        temp_dir = abspath(temp_dir)
        self.name = "{}_{}_{}".format(model.graph.name,
                                      model.model_version,
                                      model.domain.replace('.', '-'))

        visitor = HalideGraphVisitor(temp_dir=temp_dir)
        self.initializer_data = {}
        for init in model.graph.initializer:
            dims = list(init.dims)
            if init.raw_data:
                onnx_data = np.frombuffer(init.raw_data,
                                          count=np.prod(dims),
                                          dtype=from_onnx_t(init.data_type).np)
                self.initializer_data[init.name] = onnx_data

        value_info = {i.name: i.type for i in list(model.graph.input) +
                      list(model.graph.output) + list(model.graph.value_info)}

        code, objects, headers = visitor.visit(model.graph, value_info)

        code = ["#include {}".format(h) for h in headers] + \
               code

        src = '\n'.join(code)

        self.model      = model
        self.value_info = value_info
        self.temp_dir   = temp_dir
        self.headers    = headers
        self.library    = Environment.compile_library(src, objects, self.temp_dir)

    def run(self, inputs: List[ndarray], **kwargs) -> List[ndarray]:
        code = []
        args = []
        for i, ip in enumerate(list(self.model.graph.input)):
            name = ip.name
            array = self.initializer_data[name] if name in self.initializer_data else inputs[i]
            vi = VI(self.value_info[name])

            raw_file = join(self.temp_dir, "{}.raw".format(name))
            array.tofile(raw_file)

            code.extend([
                "{} v_{}[{}];".format(vi.t.c,
                                      name,
                                      '*'.join(map(str, vi.shape)) if vi.shape else 1),
                "FILE *{}_f = fopen(\"{}\", \"rb\");".format(name, raw_file),
                "fread(&v_{0}, sizeof(v_{0}), 1, {0}_f);".format(name),
                "fclose({}_f);".format(name),
                ""])
            args.append("v_" + name)

        for o, op in enumerate(list(self.model.graph.output)):
            name = op.name
            vi = VI(self.value_info[name])
            code.extend([
                "{} v_{}[{}];".format(vi.t.c,
                                      name,
                                      '*'.join(map(str, vi.shape)) if vi.shape else "1"),
                ""])
            args.append("v_" + name)

        code.extend(["{}({});".format(self.model.graph.name, ','.join(args)), ""])


        for o, op in enumerate(list(self.model.graph.output)):
            name = op.name
            vi = VI(self.value_info[name])
            raw_file = join(self.temp_dir, "{}.raw".format(name))

            # For some reason I can't create files from within pk
            if not os.path.exists(raw_file):
                f = open(raw_file, 'w')
                f.close()

            code.extend([
                "FILE *{}_f = fopen(\"{}\", \"wb\");".format(name, raw_file),
                "fwrite(&v_{0}, sizeof(v_{0}), 1, {0}_f);".format(name),
                "fclose({}_f);".format(name),
                ""])
        code.append("return 0;")

        code = ["#include {}".format(h) for h in self.headers] + \
               ["int main(int argc, char** argv)"] + \
               ["{"] + \
               ["  " + c for c in code] + \
               ["};"]

        src = '\n'.join(code)
        Environment.run_model(src, self.library, self.temp_dir)

        ret = []
        for op in list(self.model.graph.output):
            name = op.name
            vi = VI(self.value_info[name])
            raw_file = join(self.temp_dir, "{}.raw".format(name))
            ret.append(np.fromfile(raw_file, vi.t.np, -1).reshape(vi.shape))

        return ret
