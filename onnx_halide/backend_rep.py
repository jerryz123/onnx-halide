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


class HalideBackendRep(BackendRep):
    def __init__(self, model, temp_dir="build", visitor=HalideGraphVisitor):
        temp_dir = abspath(temp_dir)
        self.name = "{}_{}_{}".format(model.graph.name,
                                      model.model_version,
                                      model.domain.replace('.', '-'))

        visitor = HalideGraphVisitor(temp_dir=temp_dir)

        for init in model.graph.initializer:
            dims = list(init.dims)
            if init.raw_data:
                onnx_data = np.frombuffer(init.raw_data,
                                          count=np.prod(dims),
                                          dtype=from_onnx_t(init.data_type).np)


        value_info = {i.name: i.type for i in list(model.graph.input) +
                      list(model.graph.output) + list(model.graph.value_info)}

        code, objects, headers = visitor.visit(model.graph, value_info)

        code = ["#include {}".format(h) for h in headers] + code

        src = '\n'.join(code)
        src_cname = join(temp_dir, "generated.c")
        src_oname = join(temp_dir, "generated.o")
        src_aname = join(temp_dir, "generated.a")
        with open(src_cname, 'w') as f:
            f.write(src)

        cmd  = "riscv64-unknown-elf-g++ -std=c++11 "
        cmd += "-I./ -fno-rtti "
        cmd += "-march=rv64imafdc -mabi=lp64 "
        cmd += "-c {} -o {} ".format(src_cname, src_oname)

        r = subprocess.run(cmd, check=True, shell=True)

        cmd  = "riscv64-unknown-elf-ar rcs {} {}".format(
            src_aname,
            ' '.join([src_oname] + list(objects)))
        r = subprocess.run(cmd, check=True, shell=True)

        self.model      = model
        self.value_info = value_info
        self.temp_dir   = temp_dir
        self.headers    = headers
        self.library    = src_aname

    def run(self, inputs, **kwargs):
        code = []
        args = []
        for i, ip in enumerate(list(self.model.graph.input)):
            name = ip.name
            array = inputs[i]
            vi = VI(self.value_info[name])

            raw_file = join(self.temp_dir, "{}.raw".format(name))
            array.tofile(raw_file)

            code.extend([
                "{} v_{}[{}];".format(vi.t.c,
                                    name,
                                    '*'.join(map(str, vi.shape))),
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
                                    '*'.join(map(str, vi.shape))),
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
        src_fname = join(self.temp_dir, "main.c")
        src_bname = join(self.temp_dir, "main.riscv")
        with open(src_fname, 'w') as f:
            f.write(src)

        cmd  = "riscv64-unknown-elf-g++ -std=c++11 "
        cmd += "-fno-rtti "
        cmd += "-march=rv64imafdc -mabi=lp64 "
        cmd += "{} {} -o {} ".format(src_fname, self.library, src_bname)

        r = subprocess.run(cmd, check=True, shell=True)

        cmd  = "spike pk {}".format(src_bname)
        r = subprocess.run(cmd, check=True, shell=True)

        ret = []
        for op in list(self.model.graph.output):
            name = op.name
            vi = VI(self.value_info[name])
            raw_file = join(self.temp_dir, "{}.raw".format(name))
            ret.append(np.fromfile(raw_file, vi.t.np, -1).reshape(vi.shape))

        return ret
