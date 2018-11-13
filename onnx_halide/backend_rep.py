from onnx.backend.base import BackendRep
from onnx import TensorProto
import subprocess
import ctypes
import _ctypes
import numpy as np
import importlib
import os
from math import floor, ceil

HALIDE_DIR = "/home/jerry/Projects/Halide"

ONNX_TYPE_DECODER = lambda x:{k: v for (v, k) in TensorProto.DataType.items()}[x]
C_TYPE_DECODER = lambda x: {"FLOAT" : "float",
                            "BOOL"  : "int8_t",
                            "INT64" : "int64_t"}\
                 [ONNX_TYPE_DECODER(x)]

NP_TYPE_DECODER = lambda x: {"float"   : np.float32,
                             "int8_t"  : np.bool,
                             "int64_t" : np.int64}[x]

CTYPE_TYPE_DECODER = lambda x: {"float"   : ctypes.c_float,
                                "int8_t"  : ctypes.c_char,
                                "int64_t" : ctypes.c_longlong}[x]


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

def is_loaded(lib):
    libp = os.path.abspath(lib)
    ret = os.system("lsof -w -p %d | grep %s > /dev/null" % (os.getpid(), libp))
    return ret == 0

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

        print(model)

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
                output_strs.append("cast<{}>({}).realize({});".format(
                    c_type,
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
        for tensor, _ in outputs:
            func_name = self.funcs[tensor.name].name
            buf_name = self.buffers[tensor.name].name
            buf_shape = self.buffers[tensor.name].shape
            buf_type = self.buffers[tensor.name].type
            self.generate_func("{}_casted".format(func_name))
            self.cpp("{", 1)
            dim_vars = self.generate_dim_vars(len(buf_shape))
            self.cpp("{}_casted({}) = cast<{}>({}({}));".format(
                func_name, ",".join(dim_vars),
                buf_type,
                func_name, ",".join(dim_vars)))
            self.cpp("};", -1)
            self.cpp("{}_casted.realize({});".format(func_name, buf_name))
        # for out_str in output_strs:
        #     self.cpp(out_str)        self.cpp()
        self.cpp("};", -1)


        with open("halonet.cc", 'w') as f:
            f.write(self.halide_str)
        r = subprocess.run(["llvm-config", "--cxxflags", "--ldflags", "--libs"],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        llvm_flags = r.stdout.decode().replace('\n', ' ')

        cmd  = "g++ -g -fPIC -xc++ -ldl -lpthread -lz -lterminfo "
        cmd += "-c halonet.cc -o halonet.o "
        cmd += "{} -Wno-pedantic ".format(llvm_flags)
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


        self.halolib = ctypes.CDLL("./lib{}.so".format(self.model_name))

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

    def generate_dim_vars(self, n_vars):
        dim_vars = ["d_{}".format(i) for i in range(n_vars)]
        for dim_var in dim_vars:
            self.generate_var(dim_var)
        return dim_vars

    def generate_unary_expr(self, node, expr):
        ip_fn = self.funcs[node.input[0]].name
        op_fn = self.funcs[node.output[0]].name
        dims  = self.funcs[node.input[0]].shape
        dim_vars = self.generate_dim_vars(len(dims))
        self.cpp("{0}({2}) = {3}({1}({2}));".format(
            op_fn, ip_fn, ','.join(dim_vars), expr))

    def generate_bin_expr(self, node, expr):
        ip0_fn = self.funcs[node.input[0]]
        ip1_fn = self.funcs[node.input[1]]
        op_fn  = self.funcs[node.output[0]]
        ip0_dim = ip0_fn.shape
        ip1_dim = ip1_fn.shape
        dims = max(len(ip0_dim), len(ip1_dim))
        dim_vars = self.generate_dim_vars(dims)
        ip0_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(ip0_dim, dim_vars[-len(ip0_dim):])]
        ip1_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(ip1_dim, dim_vars[-len(ip1_dim):])]
        self.cpp("{}({}) = {}({}) {} {}({});".format(
            op_fn.name, ",".join(dim_vars[::-1]),
            ip0_fn.name, ",".join(ip0_dim_vars[::-1]),
            expr,
            ip1_fn.name, ",".join(ip1_dim_vars[::-1])))

        op_fn._shape = (None,) * dims

    def generate_red_expr(self, node, expr):
        ip_fn    = self.funcs[node.input[0]].name
        ip_shape = self.funcs[node.input[0]].shape
        op_fn    = self.funcs[node.output[0]].name

        keepdims = True
        axis = 0
        for attr in node.attribute:
            if attr.name == "keepdims":
                keepdims = attr.i == 1
            if attr.name == "axis":
                axis = attr.i

        self.cpp("RDom r(0, {});".format(ip_shape[axis]))
        dims = len(ip_shape)
        dim_vars = self.generate_dim_vars(dims)
        op_dim_vars = [dvar for i, dvar in enumerate(dim_vars)] if keepdims else \
                         [dvar for i, dvar in enumerate(dim_vars) if i != axis]
        ip_dim_vars = [(dvar if i != axis else "r") for i, dvar in enumerate(dim_vars)]
        self.cpp("{}({}) = {}({}({}))[0];".format(
            op_fn, ','.join(op_dim_vars[::-1]),
            expr,
            ip_fn, ','.join(ip_dim_vars[::-1])
            ))

    def generate_pool(self, node, expr):
        ip_fn    = self.funcs[node.input[0]].name
        op_fn    = self.funcs[node.output[0]].name
        ip_shape = self.funcs[node.input[0]].shape
        kernel_shape      = None
        count_include_pad = False
        pads              = None
        padded            = False
        auto_pad          = None
        strides           = None
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = attr.ints
            if attr.name == "count_include_pad":
                count_include_pad = attr.i == 1
            if attr.name == "pads":
                pads = [(a, b) for a, b in zip(attr.ints[::2], attr.ints[1::2])]
                padded = sum(attr.ints) > 0
            if attr.name == "auto_pad":
                auto_pad = attr.s.decode()
            if attr.name == "strides":
                strides = list(attr.ints)
        if not pads:
            if auto_pad == "SAME_UPPER":
                pads = [(floor((ks-1)/2), ceil((ks-1)/2)) for ks in kernel_shape]
                padded = True
            elif auto_pad == "SAME_LOWER":
                pads = [(ceil((ks-1)/2), floor((ks-1)/2)) for ks in kernel_shape]
                padded = True
            else:
                pads = [(0, 0) for ks in kernel_shape]
                count_include_pad = True
        if not strides:
            strides = [1 for ks in kernel_shape]
        filter_area = np.prod(kernel_shape)


        dim_vars = self.generate_dim_vars(len(ip_shape))
        self.cpp("RDom r({{{}}}, \"r\");".format(
            ','.join(["{{0,{}}}".format(i) for i in kernel_shape])))
        red_vars = ["r[{}]".format(i) for i in range(len(kernel_shape))]

        ip_vars = ["{}*{} + {} - {}".format(dv, st, rv, pad[0]) if rv else dv \
                   for (dv, rv, st, pad) in zip(dim_vars,
                                            [None] * (len(dim_vars)-len(red_vars)) + red_vars,
                                            [None] * (len(dim_vars)-len(red_vars)) + strides,
                                            [None] * (len(dim_vars)-len(red_vars)) + pads)]
        self.cpp()
        if count_include_pad:
            self.cpp("Func counts;")
            self.cpp("counts({}) = {};".format(','.join(dim_vars[::-1][:len(red_vars)]),
                                               filter_area))
        else:
            self.cpp("Func ones;")
            self.cpp("ones({}) = 1;".format(','.join(dim_vars[::-1][:len(red_vars)])))
            self.cpp("Func padded_ones = BoundaryConditions::constant_exterior(ones, 0, {{{}}});".format(
                ','.join(["{{0,{}}}".format(s) for s in ip_shape[::-1][:len(red_vars)]])))

            self.cpp("Func counts;")
            self.cpp("counts({}) = sum(padded_ones({}));".format(
                ','.join(dim_vars[::-1][:len(red_vars)]),
                ','.join(ip_vars[::-1][:len(red_vars)])
            ))

        self.cpp()
        if padded:
            self.cpp("Func padded = BoundaryConditions::constant_exterior({}, 0, {{{}}});".format(
                ip_fn,
                ','.join(["{{0,{}}}".format(s) if i < len(red_vars) else "{Expr(),Expr()}" \
                          for i, s in enumerate(ip_shape[::-1])])))
        else:
            self.cpp("Func padded = {};".format(ip_fn))
        self.cpp("{}({}) = sum({}({})) / counts({});".format(
            op_fn, ','.join(dim_vars[::-1]),
            "padded", ','.join(ip_vars[::-1]),
            ','.join(dim_vars[::-1][:len(red_vars)])))
        return

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
        elif node.op_type == "Asin":
            self.generate_unary_expr(node, "asin")
        elif node.op_type == "Atan":
            self.generate_unary_expr(node, "atan")
        elif node.op_type == "Add":
            self.generate_bin_expr(node, "+")
        elif node.op_type == "And":
            self.generate_bin_expr(node, "&")
        elif node.op_type == "Div":
            self.generate_bin_expr(node, "/")
        elif node.op_type == "ArgMax":
            self.generate_red_expr(node, "argmax")
        elif node.op_type == "ArgMin":
            self.generate_red_expr(node, "argmin")
        elif node.op_type == "AveragePool":
            self.generate_pool(node, "average")

        else:
            print()
            print("unhandled node ", node.op_type)
            print(node)
            raise NotImplementedError
        self.cpp("}", -1)
        #self.cpp("{}.realize();".format(node.output[0]))
