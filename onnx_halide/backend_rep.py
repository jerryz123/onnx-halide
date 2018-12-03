from onnx.backend.base import BackendRep
from onnx import TensorProto, shape_inference
import subprocess
import ctypes
import _ctypes
import numpy as np
import importlib
import os
from math import floor, ceil
from .tensortypes import HalogenType, HalideObj
from .generators import NodeGenerator, CppGenerator


GLOBAL_LIDX = 1 # Hack to avoid library naming collisions
if "HALIDE_DIR" in os.environ:
    HALIDE_DIR = os.environ['HALIDE_DIR']
else:
    HALIDE_DIR = "/usr/local"

JOIN_VARS = lambda vars: ','.join(vars[::-1])

CAST = lambda expr, type: "cast<{}>(Expr({}))".format(type, expr)

def is_loaded(lib):
    libp = os.path.abspath(lib)
    ret = os.system("lsof -w -p %d | grep %s > /dev/null" % (os.getpid(), libp))
    return ret == 0

class HalideBackendRep(BackendRep):
    def __init__(self, model):
        global GLOBAL_LIDX
        self.halogen_str = """"""
        self.name_map = {}
        self.model_name = "{}_{}_{}_{}".format(model.graph.name,
                                               model.model_version,
                                               model.domain.replace('.', '-'),
                                               GLOBAL_LIDX)
        GLOBAL_LIDX += 1
        self.hg = CppGenerator()
        self.sg = CppGenerator()
        self.generate_csrc(model)

    def __del__(self):
        try:
            os.remove("generated/lib{}.so".format(self.model_name))
        except FileNotFoundError:
            pass

    def run(self, inputs, **kwargs):
        i = 0
        args = []
        outputs = []
        for name, ctype in zip(self.c_args, self.halide_fn.argtypes):
            func = self.funcs[name]
            if func.is_input:
                if i >= len(inputs) or name in self.init_data:
                    input = self.init_data[name]
                else:
                    input = inputs[i]
                    if tuple(input.shape) != tuple(func.shape):
                        print(tuple(input.shape), tuple(func.shape))
                        print(func.name)
                        assert(False)
                    i += 1
                if func.is_scalar:
                    args.append(ctype(input))
                else:
                    args.append(input.ctypes.data_as(ctype))
            else:
                if func.is_scalar:
                    op = np.zeros(1, dtype=func.type.np)
                else:
                    op = np.zeros(func.shape, dtype=func.type.np)
                args.append(op.ctypes.data_as(ctype))
                outputs.append(op)
        self.halide_fn(*args)

        return outputs



    def generate_csrc(self, model):
        self.hg("#include \"Halide.h\"")
        self.hg("#include <stdint.h>")
        self.hg("#include <cfloat>")
        self.hg("#include <limits.h>")
        self.hg()

        self.funcs   = {}
        self.init_data = {}
        self.c_args = []
        # Find initial values
        for init in model.graph.initializer:
            dims = list(init.dims)
            if init.raw_data:
                onnx_data = np.frombuffer(init.raw_data,
                                          count=np.prod(dims),
                                          dtype=HalogenType.from_onnx(init.data_type).np)
            self.init_data[init.name] = onnx_data

        self.hg("using namespace Halide;")
        self.hg("using namespace BoundaryConditions;")
        self.hg("namespace {")

        self.hg("class HalOGen : public Generator<HalOGen> {", 1)
        self.hg("public:", 1)
        # Create arrays for input buffers
        input_scalars = []
        for tensor, is_output in list(map(lambda x:(x,0), model.graph.input)) \
                               + list(map(lambda x:(x,1), model.graph.output)):
            onnx_name = "f_" + tensor.name.replace('/','').replace('-','')
            c_shape = [d.dim_value for d \
                       in tensor.type.tensor_type.shape.dim]
            type  = HalogenType.from_onnx(tensor.type.tensor_type.elem_type)
            is_scalar = len(c_shape) == 0

            self.funcs[tensor.name] = HalideObj(onnx_name,
                                                tuple(c_shape),
                                                type,
                                                -1 if is_output else 1)
            if is_scalar:
                self.hg("{2}<{0}> {1}{{\"{1}\"}};".format(
                    type.c,
                    onnx_name + ("" if is_output else "_s"),
                    "Output" if is_output else "Input "))
                if not is_output:
                    input_scalars.append(self.funcs[tensor.name])
            else:
                self.hg("{4}<Buffer<{0}>> {1}{{\"{1}\", {2}}}; //{0}{3}".format(
                    type.c,
                    onnx_name,
                    len(c_shape),
                    c_shape,
                    "Output" if is_output else "Input "))

            self.c_args.append(tensor.name)



        dv = self.hg.block()
        # Generate the Halide compute function
        self.hg()
        self.hg("void generate() {", 1);

        for ip in input_scalars:
            self.hg("Func {};".format(ip.name))
            self.hg("{}() = {}_s;".format(
                ip.name, ip.name))

        # Generate Funcs for operator nodes
        generators = []
        for nidx, node in enumerate(model.graph.node):
            generators.append(NodeGenerator(node,
                                            self.hg.block(),
                                            self.hg.block(),
                                            self.funcs, self.init_data))
            self.hg("//"*10)
        n_dim_vars = max(map(lambda x:x.n_dim_vars, generators))
        dim_vars = ["d{}".format(i) for i in range(n_dim_vars)]
        for dim_var in dim_vars:
            dv("Var {};".format(dim_var))

        for op in node.output:
            if op not in self.funcs:
                f_name = "f_" + op.replace('/', '').replace('-','')
                self.generate_func(f_name)
                self.funcs[op] = HalideObj(f_name,)

        for generator in generators:
            generator.generate_alg(dim_vars[:generator.n_dim_vars])
            generator.generate_sched()


        self.hg("};", -1)
        self.hg("", -1)
        self.hg("};")
        self.hg("}", -1)
        self.hg("HALIDE_REGISTER_GENERATOR(HalOGen, halogen)")

        self.hg.write("generated/halogen_generator.cpp")

        # Generate C shim to Halide generated code
        self.halogen_str = """"""
        self.sg("#include \"Halide.h\"")
        self.sg("#include \"halogen.h\"")
        self.sg("#include <cfenv>")
        self.sg("using float16_t = Halide::float16_t;")
        self.sg("using namespace Halide::Runtime;")
        self.sg("extern \"C\" {", 1)
        py_args      = []
        buffers      = []
        ha_args      = []
        output_s     = []
        for name in self.c_args:
            fn = self.funcs[name]
            py_args.append("{0}{2} {1}".format(
                fn.type.c,
                fn.name,
                "" if fn.is_scalar and fn.is_input else "*"))
            if not fn.is_scalar:
                buffers.append("Buffer<{0}> {1}_buf({1}, {{{2}}});".format(
                    fn.type.c,
                    fn.name,
                    JOIN_VARS([str(i) for i in fn.shape])))
            elif fn.is_scalar and fn.is_output:
                buffers.append("Buffer<{0}> {1}_buf = Buffer<{0}>::make_scalar();".format(
                    fn.type.c,
                    fn.name))
                output_s.append("{0}[0] = {0}_buf();".format(fn.name))
            ha_args.append("{}{}".format(fn.name,
                                         "" if fn.is_scalar and fn.is_input else "_buf"))
        self.sg("int halogen_c({}) {{".format(','.join(py_args)), 1)

        for buf in buffers:
            self.sg(buf)
        self.sg("int r = halogen({});".format(','.join(ha_args)))

        for op in output_s:
            self.sg(op)
        self.sg("return r;")
        self.sg("}", -1)
        self.sg("}", -1)
        self.sg.write("generated/halogen_c.cpp")

        

        cmd  = "g++ -std=c++11 -fPIC -I {0}/include/ -I {0}/tools/ -g -fno-rtti "
        cmd += "generated/halogen_generator.cpp {0}/tools/GenGen.cpp {0}/lib/libHalide.a "
        cmd += "-o generated/halogen.generator -ldl -lpthread -lz -lrt -ldl -ltinfo -lm"
        cmd = cmd.format(HALIDE_DIR)
        r = subprocess.run(cmd, check=True, shell=True)

        cmd  = "ulimit -S -s 131072 ; "
        cmd += "generated/halogen.generator -g halogen -o generated -e "
        cmd += "h,static_library "
        cmd += "target=host-no_asserts"

        r = subprocess.run(cmd, check=True, shell=True)


        cmd  = "g++ -fPIC -shared -std=c++11 "
        cmd += "-I {0}/include/ -I ./generated/ "
        cmd += "generated/halogen_c.cpp generated/halogen.a "
        cmd += "{0}/lib/libHalide.a "
        cmd += "-o generated/lib{1}.so -ltinfo"
        cmd  = cmd.format(HALIDE_DIR, self.model_name)
        r = subprocess.run(cmd, check=True, shell=True)

        self.halolib = ctypes.CDLL("generated/lib{}.so".format(
            self.model_name))
        self.halide_fn = self.halolib.halogen_c
        argtypes = [self.funcs[name].type.ct_ptr \
                    if not (self.funcs[name].is_scalar and self.funcs[name].is_input) \
                    else self.funcs[name].type.ct \
                    for name in self.c_args]
        self.halide_fn.argtypes = argtypes
