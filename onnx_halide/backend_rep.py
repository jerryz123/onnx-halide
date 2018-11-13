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
C_TYPE_DECODER = lambda x: {"FLOAT16": "float16_t",
                            "FLOAT"  : "float",
                            "DOUBLE" : "double",
                            "BOOL"   : "int8_t",
                            "INT64"  : "int64_t"}\
                 [ONNX_TYPE_DECODER(x)]

NP_TYPE_DECODER = lambda x: {"float16_t": np.float16,
                             "float"    : np.float32,
                             "double"   : np.float64,
                             "int8_t"   : np.bool,
                             "int64_t"  : np.int64}[x]

CTYPE_TYPE_DECODER = lambda x: {"float16_t": ctypes.c_short,
                                "float"    : ctypes.c_float,
                                "double"   : ctypes.c_double,
                                "int8_t"   : ctypes.c_char,
                                "int64_t"  : ctypes.c_longlong}[x]


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
    def set_shape(self, shape):
        assert(not self._shape)
        self._shape = shape
    def set_type(self, type):
        assert(not self._type_str)
        self._type_str = type
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
            func = self.funcs[tensor.name]
            buf  = self.buffers[tensor.name]
            self.cpp("{}.realize({});".format(func_name, buf.name))
            if not (func.shape == buf.shape and func.type == buf.type):
                print("{}{} != {}{}".format(func.type, func.shape, buf.type, buf.shape))
                print(self.halide_str)
                exit(1)

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
        ip    = self.funcs[node.input[0]]
        op    = self.funcs[node.output[0]]

        dim_vars = self.generate_dim_vars(len(ip.shape))
        self.cpp("{0}({2}) = {3}({1}({2}));".format(
            op.name, ip.name, ','.join(dim_vars), expr))
        self.funcs[node.output[0]].set_shape(ip.shape)
        self.funcs[node.output[0]].set_type(ip.type)

    def generate_bin_expr(self, node, expr):
        ip0 = self.funcs[node.input[0]]
        ip1 = self.funcs[node.input[1]]
        op  = self.funcs[node.output[0]]
        assert(ip0.type == ip1.type)
        ip0_dim = ip0.shape
        ip1_dim = ip1.shape
        dims = max(len(ip0_dim), len(ip1_dim))
        dim_vars = self.generate_dim_vars(dims)
        ip0_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(ip0_dim, dim_vars[-len(ip0_dim):])]
        ip1_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(ip1_dim, dim_vars[-len(ip1_dim):])]
        self.cpp("{}({}) = {}({}) {} {}({});".format(
            op.name, ",".join(dim_vars[::-1]),
            ip0.name, ",".join(ip0_dim_vars[::-1]),
            expr,
            ip1.name, ",".join(ip1_dim_vars[::-1])))

        op.set_shape(
            [ip1_dim[-i] if i > len(ip0_dim) else
             (ip0_dim[-i] if i > len(ip1_dim) else
              max(ip0_dim[-i], ip1_dim[-i])) for i in range(1, dims+1)][::-1])
        op.set_type(ip0.type)


    def generate_red_expr(self, node, expr, type=None):
        ip    = self.funcs[node.input[0]]
        op    = self.funcs[node.output[0]]

        keepdims = True
        axis = 0
        for attr in node.attribute:
            if attr.name == "keepdims":
                keepdims = attr.i == 1
            if attr.name == "axis":
                axis = attr.i

        self.cpp("RDom r(0, {});".format(ip.shape[axis]))
        dims = len(ip.shape)
        dim_vars = self.generate_dim_vars(dims)
        op_dim_vars = [dvar for i, dvar in enumerate(dim_vars)] if keepdims else \
                         [dvar for i, dvar in enumerate(dim_vars) if i != axis]
        ip_dim_vars = [(dvar if i != axis else "r") for i, dvar in enumerate(dim_vars)]
        op_shape = ip.shape
        if keepdims:
            op_shape[axis] = 1
        else:
            op_shape.pop(axis)
        op_type = type if type else ip.type
        self.cpp("{}({}) = cast<{}>({}({}({}))[0]);".format(
            op.name, ','.join(op_dim_vars[::-1]),
            op_type,
            expr,
            ip.name, ','.join(ip_dim_vars[::-1])
            ))

        op.set_shape(op_shape)
        op.set_type(op_type)
    def generate_cast(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        op_type = None

        for attr in node.attribute:
            if attr.name == "to":
                op_type = C_TYPE_DECODER(attr.i)
        dim_vars = self.generate_dim_vars(len(ip.shape))
        self.cpp("{}({}) = cast<{}>({}({}));".format(
            op.name, ','.join(dim_vars[::-1]),
            op_type,
            ip.name, ','.join(dim_vars[::-1])))
        op.set_shape(ip.shape)
        op.set_type(op_type)

    def generate_bn(self, node):
        x    = self.funcs[node.input[0]]
        s    = self.funcs[node.input[1]]
        bias = self.funcs[node.input[2]]
        mean = self.funcs[node.input[3]]
        var  = self.funcs[node.input[4]]
        op   = self.funcs[node.output[0]]
        eps  = 0
        eps_t = "float"
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps   = attr.f
                eps_t = C_TYPE_DECODER(attr.type)
        #s * (x - mean) / np.sqrt(var + epsilon) + bias
        dim_vars = self.generate_dim_vars(4)
        self.cpp("Expr eps({});".format(eps))
        self.cpp("{}({}) = cast<{}>({}({})*({}({}) - {}({})) / sqrt({}({})+eps) + {}({}));".format(
            op.name, ','.join(dim_vars[::-1]),
            x.type,
            s.name, dim_vars[-3],
            x.name, ','.join(dim_vars[::-1]),
            mean.name, dim_vars[-3],
            var.name, dim_vars[-3],
            bias.name, dim_vars[-3]))
        op.set_shape(x.shape)
        op.set_type(x.type)


    def generate_pool(self, node):
        ip    = self.funcs[node.input[0]]
        op    = self.funcs[node.output[0]]
        if node.op_type == "Conv":
            w = self.funcs[node.input[1]]

        pool_shape        = None
        count_include_pad = False
        pads              = None
        padded            = False
        auto_pad          = None
        strides           = None
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                pool_shape = list(attr.ints)
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
                pads = [(floor((ks-1)/2), ceil((ks-1)/2)) for ks in pool_shape]
                padded = True
            elif auto_pad == "SAME_LOWER":
                pads = [(ceil((ks-1)/2), floor((ks-1)/2)) for ks in pool_shape]
                padded = True
            else:
                pads = [(0, 0) for ks in pool_shape]
                count_include_pad = True
        if not strides:
            strides = [1 for ks in pool_shape]
        filter_area = np.prod(pool_shape)


        dim_vars = self.generate_dim_vars(len(ip.shape))
        self.cpp("RDom r({{{}}}, \"r\");".format(
            ','.join(["{{0,{}}}".format(i) for i in pool_shape])))
        red_vars = ["r[{}]".format(i) for i in range(len(pool_shape))]

        n_ign_dims = len(dim_vars) - len(red_vars)
        ip_vars = ["{}*{}+{}-{}".format(dv, st, rv, pad[0]) if rv else dv \
                   for (dv, rv, st, pad) in zip(dim_vars,
                                            [None] * n_ign_dims + red_vars,
                                            [None] * n_ign_dims + strides,
                                            [None] * n_ign_dims + pads)]

        if padded:
            self.cpp("Func padded = BoundaryConditions::constant_exterior({}, 0, {{{}}});".format(
                ip.name,
                ','.join(["{{0,{}}}".format(s) if i < len(red_vars) else "{Expr(),Expr()}" \
                          for i, s in enumerate(ip.shape[::-1])])))
        else:
            self.cpp("Func padded = {};".format(ip.name))
        self.cpp()
        if node.op_type == "AveragePool":
            self.cpp("Func counts;")
            if count_include_pad:
                self.cpp("counts({}) = {};".format(','.join(dim_vars[::-1][:len(red_vars)]),
                                                   filter_area))
            else:
                self.cpp("Func ones;")
                self.cpp("ones({}) = 1;".format(','.join(dim_vars[::-1][:len(red_vars)])))
                self.cpp("Func padded_ones = BoundaryConditions::constant_exterior(ones, 0, {{{}}});".format(
                    ','.join(["{{0,{}}}".format(s) for s in ip.shape[::-1][:len(red_vars)]])))
                self.cpp("counts({}) = sum(padded_ones({}));".format(
                    ','.join(dim_vars[::-1][:len(red_vars)]),
                    ','.join(ip_vars[::-1][:len(red_vars)])
                ))
            self.cpp("{}({}) = sum(padded({})) / counts({});".format(
                op.name, ','.join(dim_vars[::-1]),
                ','.join(ip_vars[::-1]),
                ','.join(dim_vars[::-1][:len(red_vars)])))
        elif node.op_type == "Conv":
            n_kern_ign_dims = len(w.shape) - len(red_vars)
            kern_vars = [rv if rv else dv for (dv, rv) in zip(dim_vars[-len(w.shape):],
                                                              [None] * n_kern_ign_dims + red_vars)]

            self.cpp("{}({}) = sum(padded({}) * {}({}));".format(
                op.name, ','.join(dim_vars[::-1]),
                ','.join(ip_vars[::-1]),
                w.name, ','.join(kern_vars[::-1])
                ))


        op_shape = [floor((ips + pad[0] + pad[1] - ks) / stride + 1) if pad else ips \
                    for (ips, pad, ks, stride) \
                    in zip(ip.shape,
                           [None] * n_ign_dims + pads,
                           [None] * n_ign_dims + pool_shape,
                           [None] * n_ign_dims + strides)]
        op.set_shape(op_shape)
        op.set_type(ip.type)
        return

    def generate_node(self, nidx, node):
        for op in node.output:
            f_name = op.replace('/', '') + "_func"
            self.generate_func(f_name)
            assert(op not in self.funcs)
            self.funcs[op] = HalideObj(f_name,)
        self.cpp("{{ // {} {} {}".format(node.op_type, nidx, node.name), 1)
        if   node.op_type == "Abs":
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
            self.generate_red_expr(node, "argmax", "int64_t")
        elif node.op_type == "ArgMin":
            self.generate_red_expr(node, "argmin", "int64_t")
        elif node.op_type == "BatchNormalization":
            self.generate_bn(node)
        elif node.op_type == "AveragePool":
            self.generate_pool(node)
        elif node.op_type == "Conv":
            self.generate_pool(node)
        elif node.op_type == "Cast":
            self.generate_cast(node)
        else:
            print()
            print("unhandled node ", node.op_type)
            print(node)
            raise NotImplementedError
        self.cpp("}", -1)

        for op in node.output:
            try:
                self.cpp("// {}  {}{}".format(op, self.funcs[op].type,
                                              self.funcs[op].shape))
                self.funcs[op].type
            except:
                print(node)
                print(self.halide_str)

        #self.cpp("{}.realize();".format(node.output[0]))
