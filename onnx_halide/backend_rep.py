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


class HalideBackendRep(BackendRep):
    def __init__(self, model):

        self.halide_str = """"""
        self.indent = 0
        self.buffers = {}
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
            op = np.zeros(shape=self.buffers[op_name][1],
                          dtype=NP_TYPE_DECODER(self.buffers[op_name][2]))
            ctypes.memmove(op.ctypes.data, op_ptr, op.nbytes)
            ops.append(op)
            print(op)
            
        return ops

    def generate_csrc(self, model):
        self.cpp("#include \"Halide.h\"")
        self.cpp("#include <stdint.h>")
        self.cpp("using namespace Halide;")
        self.cpp()
        init_data = {}

        # Find initial values
        for init in model.graph.initializer:
            if init.raw_data:
                onnx_data = np.frombuffer(init.raw_data, dtype=float)
            c_arr = ", ".join([str(i) for i in onnx_data])
            init_data[init.name] = c_arr

        # Create arrays for input buffers, assign these with initial vlaues
        for ip in model.graph.input:
            c_shape = [d.dim_value for d in ip.type.tensor_type.shape.dim]
            c_type  = C_TYPE_DECODER(ip.type.tensor_type.elem_type)
            c_name  = ip.name.replace('/', '')
            c_size  = np.prod(c_shape)
            c_str   = "extern \"C\" {} c_{}[{}]".format(c_type, c_name, c_size)
            if ip.name in init_data:
                c_str += " = {{{}}}".format(init_data[ip.name])
            c_str += ";"
            self.cpp(c_str)
            self.buffers[ip.name] = (c_name, c_shape, c_type)

        # Create arrays for output buffers
        self.cpp()
        for op in model.graph.output:
            c_type  = C_TYPE_DECODER(op.type.tensor_type.elem_type)
            c_name  = "out_" + op.name.replace('/', '')
            c_shape = [d.dim_value for d in op.type.tensor_type.shape.dim]
            c_size  = np.prod(c_shape)
            self.cpp("extern \"C\" {} c_{}[{}];".format(c_type, c_name, c_size))
            self.buffers[op.name] = (c_name, c_shape, c_type)

        # Create arrays for constant nodes
        self.cpp()
        for cnode in model.graph.node:
            if cnode.op_type == "Constant":
                for (op_name, attr) in zip(cnode.output, cnode.attribute):
                    c_name  = op_name.replace('/', '')
                    c_type  = C_TYPE_DECODER(attr.t.data_type)
                    c_shape = [d for d in attr.t.dims]
                    c_size  = np.prod(c_shape)
                    if not c_shape: # Scalar const
                        if attr.t.float_data:
                            self.cpp("{} c_{} = {};".format(c_type, c_name, attr.t.float_data[0]))
                        else:
                            raise NotImplementedError
                    else:
                        if attr.t.float_data:
                            c_arr = ", ".join([str(i) for i in attr.t.float_data])
                        else:
                            raise NotImplementedError
                        self.cpp("{} c_{}[{}] = {{{}}};".format(c_type, c_name, c_size, c_arr))

        # Create Halide Buffers for the c arrays
        self.cpp()
        for n, (c_name, c_shape, c_type) in self.buffers.items():
            self.cpp("Buffer<{0}> {1}(c_{1}, {{{2}}});".format(
                c_type,
                c_name,
                ', '.join([str(i) for i in c_shape][::-1])))


        # Generate the Halide compute function
        self.cpp()
        self.cpp("extern \"C\" void halide_compute() {", 1);

        for nidx, node in enumerate(model.graph.node):
            self.generate_node(nidx, node)
            break

        for op in model.graph.output:
            self.cpp("{0}.realize(out_{0});".format(op.name))
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

        if r:
            print(r.stdout.decode())
            print(r.stderr.decode())

        cmd  = "g++ -shared -o lib{}.so halonet.o {}/lib/libHalide.a ".format(
            self.model_name,
            HALIDE_DIR)
        cmd += "-ldl -lz -ltinfo -lpthread"

        r = subprocess.run(cmd, shell=True)


        self.halolib = ctypes.CDLL("./lib{}.so".format(self.model_name),
                                   mode=ctypes.RTLD_GLOBAL)

        self.in_pointers = []
        for ip in model.graph.input:
            ctype_type = CTYPE_TYPE_DECODER(self.buffers[ip.name][2])
            ip_ptr = ctypes.pointer(ctype_type.in_dll(self.halolib,
                                                      "c_" + ip.name))
            self.in_pointers.append(ip_ptr)

        self.out_pointers = []
        for op in model.graph.output:
            ctype_type = CTYPE_TYPE_DECODER(self.buffers[op.name][2])
            op_ptr = ctypes.pointer(ctype_type.in_dll(self.halolib,
                                                      "c_out_" + op.name))
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
        ip = node.input[0]
        op = node.output[0]
        dims = self.buffers[ip][1]
        dim_vars = ["d_{}".format(i) for i in range(len(dims))]
        for dim_var in dim_vars:
            self.generate_var(dim_var)
        self.cpp("{0}({2}) = {3}({1}({2}));".format(
            op, ip, ','.join(dim_vars), expr))

    def generate_bin_expr(self, node, expr):
        ip0 = node.input[0]
        ip1 = node.input[1]
        op = node.output[0]
        ip0_dim = self.buffers[ip0][1]
        ip1_dim = self.buffers[ip1][1]
        out_dim = self.buffers[op][1]
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
            self.generate_func(op)
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
