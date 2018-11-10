from onnx.backend.base import BackendRep
from onnx import TensorProto
import subprocess
import numpy as np

HALIDE_DIR = "/home/jerry/Projects/Halide"

C_TYPE_DECODER = lambda x: {"FLOAT" : "float"}[{k: v for (v, k) in TensorProto.DataType.items()}[x]]

class HalideBackendRep(BackendRep):
    def __init__(self, model):

        self.halide_str = """
#include "Halide.h"
using namespace Halide;

"""
        indent = 0
        def cpp(s, incr=0):
            nonlocal indent
            if incr < 0:
                indent += incr
            self.halide_str += "{}{}\n".format(' ' * indent, s)
            if incr > 0:
                indent += incr

        buffers = {}

        model_name = "{}_{}_{}".format(model.graph.name, model.model_version,
                                       model.domain.replace('.', '-'))

        init_data = {}
        for init in model.graph.initializer:
            if init.raw_data:
                onnx_data = np.frombuffer(init.raw_data, dtype=float)
            c_arr = ", ".join([str(i) for i in onnx_data])
            init_data[init.name] = c_arr


        for ip in model.graph.input:
            c_type = C_TYPE_DECODER(ip.type.tensor_type.elem_type)
            c_name = ip.name.replace('/', '')
            c_size = np.prod([d.dim_value for d in ip.type.tensor_type.shape.dim])
            c_str = "{} {}[{}]".format(c_type, c_name, c_size)
            if ip.name in init_data:
                c_str += " = {{{}}}".format(init_data[ip.name])
            c_str += ";"
            cpp(c_str)
            buffers[ip.name] = (c_name)

        for op in model.graph.output:
            c_type = C_TYPE_DECODER(op.type.tensor_type.elem_type)
            c_name = op.name.replace('/', '')
            c_size = np.prod([d.dim_value for d in op.type.tensor_type.shape.dim])
            cpp("{} {}[{}];".format(c_type, c_name, c_size))

        func_str = "void halide_compute() {\n"

        cpp("int main(){", 1)
        cpp("printf(\"%.3f\\n\", data_0[0]);")
        cpp("};", -1)
        with open("halonet.cc", 'w') as f:
            f.write(self.halide_str)
        r = subprocess.run(["llvm-config", "--cxxflags", "--ldflags", "--libs"],
                           stdout=subprocess.PIPE)
        llvm_flags = r.stdout.decode().replace('\n', ' ')
        cmd  = "g++ -g -Wall -xc++ - "
        cmd += "-o halonet "
        cmd += "{} ".format(llvm_flags)
        cmd += "-lHalide -I{0}/include -L{0}/lib ".format(HALIDE_DIR)
        print(cmd)
        r = subprocess.run(cmd,
                           shell=True,
                           input=self.halide_str.encode())
        exit(0)
