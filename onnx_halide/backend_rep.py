from onnx.backend.base import BackendRep
from onnx import TensorProto
import subprocess
import ctypes
import _ctypes
import numpy as np
import os

HALIDE_DIR = "/home/jerry/Projects/Halide"

ONNX_TYPE_DECODER = lambda x:{k: v for (v, k) in TensorProto.DataType.items()}[x]
C_TYPE_DECODER = lambda x: {"FLOAT" : "float",
                            "BOOL"  : "int8_t"}\
                 [ONNX_TYPE_DECODER(x)]

NP_TYPE_DECODER = lambda x: {"float" : np.float32,
                             "bool"  : np.bool}[x]

CTYPE_TYPE_DECODER = lambda x: {"float" : ctypes.c_float,
                                "int8_t"  : ctypes.c_char}[x]


class HalideObj:
    def __init__(self, name=None, shape=None, type_str=None):
        self._name = name
        self._shape = shape
        self._type_str = type_str
    @property
    def name(self):
        assert(self._name)
        return self._name
    @property
    def shape(self):
        assert(self._shape)
        return self._shape
    @property
    def type(self):
        assert(self._type_str)
        return self._type_str

    def __repr__(self):
        return "({}, {}, {})".format(self._name,
                                     self._shape,
                                     self._type_str)

class HalideBackendRep(BackendRep):
    def __init__(self, model):

        self.halide_str = """"""
        self.indent = 0
        self.buffers = {}
        self.name_map = {}
        self.model_name = "{}_{}_{}".format(model.graph.name, model.model_version,
                                       model.domain.replace('.', '-'))
        self.generate_csrc(model)

    def cpp(self, s="", incr=0):
        if incr < 0:
            self.indent += incr
        self.halide_str += "{}{}\n".format("  " * self.indent, s)
        if incr > 0:
            self.indent += incr

    def run(self, inputs, **kwargs):
        print()
        for ip, ip_ptr in zip(inputs, self.in_pointers):
            print(ip)
            print()
            ctypes.memmove(ip_ptr, ip.ctypes.data, ip.nbytes)
        self.halide_fn()
        ops = []
        for op_name, op_ptr in self.out_pointers:
            op = np.zeros(shape=self.buffers[op_name].shape,
                          dtype=NP_TYPE_DECODER(self.buffers[op_name].type))
            ctypes.memmove(op.ctypes.data, op_ptr, op.nbytes)
            ops.append(op)
            print(op)

        return ops

    def generate_csrc(self, model):
        self.cpp("#include \"Halide.h\"")
        self.cpp("#include <stdint.h>")
        self.cpp("using namespace Halide;")
        self.cpp()

        self.arrays  = {}
        self.buffers = {}
        self.funcs   = {}
        init_data = {}

        # Find initial values
        for init in model.graph.initializer:
            if init.raw_data:
                onnx_data = np.frombuffer(init.raw_data, dtype=float)
            c_arr = ", ".join([str(i) for i in onnx_data])
            init_data[init.name] = c_arr

        # Create arrays for input buffers, assign these with initial values
        func_strs   = []
        output_strs = []
        inputs      = [(ip, 1) for ip in model.graph.input]
        outputs     = [(op, 0) for op in model.graph.output]
        for idx, (tensor, is_ip) in enumerate(inputs + outputs):
            onnx_name = tensor.name.replace('/', '')
            io = "in" if is_ip else "out"

            c_shape = [d.dim_value for d in tensor.type.tensor_type.shape.dim]
            c_type  = C_TYPE_DECODER(tensor.type.tensor_type.elem_type)
            c_name  = "{}_{}_c".format(onnx_name, io)
            c_size  = np.prod(c_shape)
            c_str   = "{} {}[{}]".format(c_type, c_name, c_size)
            if tensor.name in init_data:
                c_str += " = {{{}}}".format(init_data[tensor.name])
            c_str += ";"
            self.cpp(c_str)
            self.arrays[tensor.name] = HalideObj(c_name, c_shape, c_type)

            buf_name = "{}_{}_buf".format(onnx_name, io)
            self.cpp("Buffer<{0}> {3}({1}, {{{2}}});".format(
                c_type,
                c_name,
                ', '.join([str(i) for i in c_shape][::-1]),
                buf_name))
            self.buffers[tensor.name] = HalideObj(buf_name, c_shape, c_type)

            func_name = "{}_func".format(onnx_name, io)
            if is_ip:
                func_strs.append("Func {}({});".format(func_name, buf_name))
                self.funcs[tensor.name] = HalideObj(func_name, c_shape, c_type)
            else:
                output_strs.append("{}.realize({});".format(
                    func_name, buf_name))
            self.cpp()

        # # Create arrays for constant nodes
        # self.cpp()
        # for cnode in model.graph.node:
        #     if cnode.op_type == "Constant":
        #         for (op_name, attr) in zip(cnode.output, cnode.attribute):
        #             c_name  = op_name.replace('/', '')
        #             c_type  = C_TYPE_DECODER(attr.t.data_type)
        #             c_shape = [d for d in attr.t.dims]
        #             c_size  = np.prod(c_shape)
        #             if not c_shape: # Scalar const
        #                 if attr.t.float_data:
        #                     self.cpp("{} c_{} = {};".format(c_type, c_name, attr.t.float_data[0]))
        #                 else:
        #                     raise NotImplementedError
        #             else:
        #                 if attr.t.float_data:
        #                     c_arr = ", ".join([str(i) for i in attr.t.float_data])
        #                 else:
        #                     raise NotImplementedError
        #                 self.cpp("{} c_{}[{}] = {{{}}};".format(c_type, c_name, c_size, c_arr))


        # Generate the Halide compute function
        self.cpp()
        self.cpp("extern \"C\" void halide_compute() {", 1);

        # Generate funcs for input buffers
        for func_str in func_strs:
            self.cpp(func_str)
        self.cpp()

        # Generate Funcs for operator nodes
        for nidx, node in enumerate(model.graph.node):
            self.generate_node(nidx, node)
        self.cpp()

        # Realize the output funcs into output buffesr
        for out_str in output_strs:
            self.cpp(out_str)
        self.cpp()
        self.cpp("};", -1)


        with open("halonet.cc", 'w') as f:
            f.write(self.halide_str)
        r = subprocess.run(["llvm-config", "--cxxflags", "--ldflags", "--libs"],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        llvm_flags = r.stdout.decode().replace('\n', ' ')

        cmd  = "g++ -g -fPIC -xc++ -ldl -lpthread -lz -lterminfo "
        cmd += "-c halonet.cc -o halonet.o "
        cmd += "{} ".format(llvm_flags)
        cmd += "-lHalide -I{0}/include -L{0}/lib ".format(HALIDE_DIR)

        r = subprocess.run(cmd,
                           shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        print(r.stdout.decode())
        print(r.stderr.decode())

        cmd  = "g++ -shared -o lib{}.so halonet.o {}/lib/libHalide.a ".format(
            self.model_name,
            HALIDE_DIR)
        cmd += "-ldl -lz -ltinfo -lpthread"

        r = subprocess.run(cmd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           shell=True)
        print(r.stdout.decode())
        print(r.stderr.decode())


        self.halolib = ctypes.CDLL("./lib{}.so".format(self.model_name),
                                   mode=ctypes.RTLD_GLOBAL)

        self.in_pointers = []
        for ip in model.graph.input:
            ctype_type = CTYPE_TYPE_DECODER(self.arrays[ip.name].type)
            ip_ptr = ctypes.pointer(ctype_type.in_dll(self.halolib,
                                                      self.arrays[ip.name].name))
            self.in_pointers.append(ip_ptr)

        self.out_pointers = []
        for op in model.graph.output:
            ctype_type = CTYPE_TYPE_DECODER(self.arrays[op.name].type)
            op_ptr = ctypes.pointer(ctype_type.in_dll(self.halolib,
                                                      self.arrays[op.name].name))
            self.out_pointers.append((op.name, op_ptr))

        self.halide_fn = self.halolib.halide_compute


    def generate_var(self, var):
        self.cpp("Var {0}(\"{0}\");".format(var))

    def generate_func(self, fname):
        self.cpp("Func {0}(\"{0}\");".format(fname))


    def generate_conv(self, node):
        op = node.output[0]

        ip, weight = node.input[0], node.input[1]
        bias = node.input[1] if len(node.input) == 3 else None

        stride_x, stride_y = 1, 1
        pad_x0, pad_x1, pad_y0, pad_y1 = 0, 0, 0, 0

        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape_x, kernel_shape_y = attr.ints
            elif attr.name == "pads":
                pad_x0, pad_x1, pad_y0, pad_y1 = attr.ints
            elif attr.name == "strides":
                stride_x, stride_y = attr.ints


        self.generate_var("n_v")
        self.generate_var("c_v")
        self.generate_var("x_v")
        self.generate_var("y_v")

        self.cpp("int  n_in = {}.dim(0).extent();".format(ip))
        self.cpp("int ch_in = {}.dim(1).extent();".format(ip))
        self.cpp("int  h_in = {}.dim(2).extent();".format(ip))
        self.cpp("int  w_in = {}.dim(3).extent();".format(ip))
        self.cpp()
        self.generate_func("in_bounded")
        self.cpp("in_bounded = BoundaryConditions::constant_exterior(", 1)
        self.cpp("{}, 0);".format(ip))
        self.cpp("", -1)
        self.cpp("RDom r(0,ch_in, 0,{}, 0,{});".format(kernel_shape_x, kernel_shape_y))
        self.cpp()
        if bias:
            self.cpp("{}(n_v,c_v,x_v,y_v) = {}(c_v);".format(op, bias))
        self.cpp()
        self.cpp("{}(n_v,c_v,x_v,y_v) += {}(n_v,r[0],r[1],r[2]) \
* in_bounded(n_v,r[0],x_v*{}+r[1],y_v*{}+r[2]);".format(
    op, weight, stride_x, stride_y))

    def generate_unary_expr(self, node, expr):
        print(self.funcs)
        ip_fn = self.funcs[node.input[0]].name
        op_fn = self.funcs[node.output[0]].name
        dims  = self.funcs[node.input[0]].shape
        dim_vars = ["d_{}".format(i) for i in range(len(dims))]
        for dim_var in dim_vars:
            self.generate_var(dim_var)
        self.cpp("{0}({2}) = {3}({1}({2}));".format(
            op_fn, ip_fn, ','.join(dim_vars), expr))

    def generate_bin_expr(self, node, expr):
        ip0 = node.input[0]
        ip1 = node.input[1]
        op = node.output[0]
        ip0_dim = self.buffers[ip0].shape
        ip1_dim = self.buffers[ip1].shape
        out_dim = self.buffers[op].shape
        dim_vars = ["d_{}".format(i) for i in range(len(out_dim))]
        for dim_var in dim_vars:
            self.generate_var(dim_var)
        self.cpp("{}({}) = {}({}) {} {}({});".format(
            op, ",".join(dim_vars),
            ip0, ",".join(dim_vars[:len(ip0_dim)]),
            expr,
            ip1, ",".join(dim_vars[:len(ip1_dim)])))

    def generate_node(self, nidx, node):
        for op in node.output:
            f_name = op.replace('/', '') + "_func"
            self.generate_func(f_name)
            assert(op not in self.funcs)
            self.funcs[op] = HalideObj(f_name,)
        self.cpp("{{ // {} {} {}".format(node.op_type, nidx, node.name), 1)
        if node.op_type == "Conv":
            self.generate_conv(node)
        elif node.op_type == "Abs":
            self.generate_unary_expr(node, "abs")
        elif node.op_type == "Acos":
            self.generate_unary_expr(node, "acos")
        elif node.op_type == "Add":
            self.generate_bin_expr(node, "+")
        elif node.op_type == "And":
            self.generate_bin_expr(node, "&")
        elif node.op_type == "Div":
            a = 1

        else:
            print("unhandled node ", node.op_type)
            raise NotImplementedError
        self.cpp("}", -1)
        #self.cpp("{}.realize();".format(node.output[0]))
