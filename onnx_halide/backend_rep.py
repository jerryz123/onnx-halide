from onnx.backend.base import BackendRep
from onnx import TensorProto, shape_inference
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
                            "INT32"  : "int32_t",
                            "INT64"  : "int64_t"}\
                 [ONNX_TYPE_DECODER(x)]

NP_TYPE_DECODER = lambda x: {"float16_t": np.float16,
                             "float"    : np.float32,
                             "double"   : np.float64,
                             "int8_t"   : np.bool,
                             "int32_t"  : np.int32,
                             "int64_t"  : np.int64}[x]

CTYPE_TYPE_DECODER = lambda x: {"float16_t": ctypes.c_short,
                                "float"    : ctypes.c_float,
                                "double"   : ctypes.c_double,
                                "int8_t"   : ctypes.c_char,
                                "int32_t"  : ctypes.c_int,
                                "int64_t"  : ctypes.c_longlong}[x]

MIN_TYPE_DECODER = lambda x: {"float16_t": "float16_t.make_infinity(0)",
                              "float"    : "cast<float>(Expr(-FLT_MAX))",
                              "double"   : "cast<double>(Expr(-DBL_MAX))",
                              "int8_t"   : "cast<int8_t>(Expr(-CHAR_MAX))",
                              "int32_t"   : "cast<int8_t>(Expr(-INT_MAX))",
                              "int64_t"  : "cast<int64_t>(Expr(-LLONG_MAX))"}[x]
MAX_TYPE_DECODER = lambda x: {"float16_t": "float16_t.make_infinity(1)",
                              "float"    : "cast<float>(Expr(FLT_MAX))",
                              "double"   : "cast<double>(Expr(DBL_MAX))",
                              "int8_t"   : "cast<int8_t>(Expr(CHAR_MAX))",
                              "int32_t"   : "cast<int8_t>(Expr(INT_MAX))",
                              "int64_t"  : "cast<int64_t>(Expr(LLONG_MAX))"}[x]

JOIN_VARS = lambda vars: ','.join(vars[::-1])

class OnnxAttr:
    def __init__(self, attr, v_fn=lambda x:x, value=None, type=None):
        self.value = value
        self.type = type
        if hasattr(attr, "i"):
            self.value = attr.i
        elif hasattr(attr, "f"):
            self.value = attr.f
        if hasattr(attr, "type"):
            self.type = attr.type
        self.value = v_fn(self.value)
class OnnxAttrs:
    def __init__(self, attrs, **kwargs):
        for k, v in kwargs.items():
            if v:
                setattr(self, k, OnnxAttr(None, v[0], v[1]))
            else:
                setattr(self, k, None)
        for attr in attrs:
            if hasattr(self, attr.name):
                setattr(self, attr.name, OnnxAttr(attr))

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
        self.model_name = "{}_{}_{}".format(model.graph.name,
                                            model.model_version,
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
                          dtype=NP_TYPE_DECODER(
                              self.buffers[op_name].type))
            ctypes.memmove(op.ctypes.data, op_ptr, op.nbytes)
            ops.append(op)
            print(op)

        return ops



    def generate_csrc(self, model):


        self.cpp("#include \"Halide.h\"")
        self.cpp("#include <stdint.h>")
        self.cpp("#include <cfloat>")
        self.cpp("#include <limits.h>")
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

            c_shape = [d.dim_value for d \
                       in tensor.type.tensor_type.shape.dim]
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
            self.buffers[tensor.name] = HalideObj(buf_name,
                                                  c_shape,
                                                  c_type)

            func_name = "{}_func".format(onnx_name, io)
            if is_ip:
                func_strs.append("Func {}({});".format(func_name,
                                                       buf_name))
                self.funcs[tensor.name] = HalideObj(func_name,
                                                    c_shape,
                                                    c_type)
            else:
                output_strs.append("cast<{}>({}).realize({});".format(
                    c_type,
                    func_name, buf_name))
            self.cpp()

        # Create arrays for constant nodes
        self.cpp()
        for cnode in model.graph.node:
            if cnode.op_type == "Constant":
                tensor_name = cnode.output[0]
                attr = cnode.attribute[0]
                onnx_name = tensor_name.replace('/', '')

                c_shape = attr.t.dims
                c_type  = C_TYPE_DECODER(attr.t.data_type)
                c_name  = "{}_constant_c".format(onnx_name)
                c_size  = np.prod(c_shape)

                if attr.t.float_data:
                    init_data = ','.join(map(str, attr.t.float_data))
                
                self.cpp("{} {}[{}] = {{{}}};".format(c_type,
                                                      c_name,
                                                      c_size,
                                                      init_data))
                self.arrays[tensor_name] = HalideObj(c_name,
                                                     c_shape,
                                                     c_type)

                buf_name = "{}_constant_buf".format(onnx_name)
                self.cpp("Buffer<{0}> {3}({1}, {{{2}}});".format(
                    c_type,
                    c_name,
                    ', '.join([str(i) for i in c_shape][::-1]),
                    buf_name))
                self.buffers[tensor_name] = HalideObj(buf_name,
                                                      c_shape, c_type)

                func_name = "{}_func".format(onnx_name, io)
                func_strs.append("Func {}({});".format(func_name,
                                                       buf_name))
                self.funcs[tensor_name] = HalideObj(func_name,
                                                    c_shape, c_type)


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

        # Realize the output funcs into output buffers
        for tensor, _ in outputs:
            func = self.funcs[tensor.name]
            buf  = self.buffers[tensor.name]
            self.cpp("{}.realize({});".format(func_name, buf.name))
            if not (func.shape == buf.shape and func.type == buf.type):
                print("{}{} != {}{}".format(func.type, func.shape,
                                            buf.type, buf.shape))
                print(self.halide_str)
                print(model)
                exit(1)

        # for out_str in output_strs:
        #     self.cpp(out_str)        self.cpp()
        self.cpp("};", -1)


        with open("halonet.cc", 'w') as f:
            f.write(self.halide_str)
        r = subprocess.run(["llvm-config", "--cxxflags",
                            "--ldflags", "--libs"],
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
            ip_ptr = ctypes.pointer(ctype_type.in_dll(
                self.halolib,
                self.arrays[ip.name].name))
            self.in_pointers.append(ip_ptr)

        self.out_pointers = []
        for op in model.graph.output:
            ctype_type = CTYPE_TYPE_DECODER(self.arrays[op.name].type)
            op_ptr = ctypes.pointer(ctype_type.in_dll(
                self.halolib,
                self.arrays[op.name].name))
            self.out_pointers.append((op.name, op_ptr))

        self.halide_fn = self.halolib.halide_compute


    def generate_var(self, var):
        self.cpp("Var {0}(\"{0}\");".format(var))

    def generate_func(self, fname):
        self.cpp("Func {0}(\"{0}\");".format(fname))

    def generate_dim_vars(self, n_vars):
        dim_vars = ["d_{}".format(i) for i in range(n_vars)]
        for dim_var in dim_vars:
            self.generate_var(dim_var)
        return dim_vars

    def generate_rdom(self, shape):
        self.cpp("RDom r({});".format(','.join(["0,{}".format(s) \
                                                for s in shape])))
        return ["r[{}]".format(i) for i in range(len(shape))]

    def generate_unary_expr(self, node, expr):
        ip      = self.funcs[node.input[0]]
        op      = self.funcs[node.output[0]]
        op_type = ip.type
        min_v   = MIN_TYPE_DECODER(op_type)
        max_v   = MAX_TYPE_DECODER(op_type)
        alpha   = 1.0

        for attr in node.attribute:
            if attr.name == "alpha":
                alpha = attr.f
            if attr.name == "to":
                op_type = C_TYPE_DECODER(attr.i)
            if attr.name == "max":
                max_v = "Expr({})".format(attr.f)
            if attr.name == "min":
                min_v = "Expr({})".format(attr.f)
        dim_vars = self.generate_dim_vars(len(ip.shape))

        ip_expr = "{}({})".format(ip.name, ','.join(dim_vars[::-1]))

        if node.op_type == "Cast":
            expr = expr.format(ip_expr, op_type)
        elif node.op_type == "Clip":
            expr = expr.format(ip_expr, min_v, max_v)
        elif node.op_type == "Elu":
            expr = expr.format(ip_expr, alpha, ip.type)
        else:
            expr = expr.format(ip_expr)


        self.cpp("{}({}) = {};".format(
            op.name, ','.join(dim_vars[::-1]),
            expr))
        op.set_shape(ip.shape)
        op.set_type(op_type)

    def generate_bin_expr(self, node, expr, op_type=None):
        ip0 = self.funcs[node.input[0]]
        ip1 = self.funcs[node.input[1]]
        op  = self.funcs[node.output[0]]

        if not op_type:
            op_type = ip0.type
        assert(ip0.type == ip1.type)
        ip0_dim = ip0.shape
        ip1_dim = ip1.shape
        dims = max(len(ip0_dim), len(ip1_dim))
        dim_vars = self.generate_dim_vars(dims)
        ip0_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(ip0_dim, dim_vars[-len(ip0_dim):])]
        ip1_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(ip1_dim, dim_vars[-len(ip1_dim):])]



        self.cpp("{}({}) = cast<{}>({}({}) {} {}({}));".format(
            op.name, ",".join(dim_vars[::-1]),
            op_type,
            ip0.name, ",".join(ip0_dim_vars[::-1]),
            expr,
            ip1.name, ",".join(ip1_dim_vars[::-1])))

        op.set_shape(
            [ip1_dim[-i] if i > len(ip0_dim) else
             (ip0_dim[-i] if i > len(ip1_dim) else
              max(ip0_dim[-i], ip1_dim[-i])) \
             for i in range(1, dims+1)][::-1])
        op.set_type(op_type)

    def generate_flatten(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        axis = 1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        op_shape = ip.shape
        dim_vars = self.generate_dim_vars(2)
        if axis == 0:
            op_shape = [1] + [np.prod(ip.shape)]
            prevs = ip.shape[1:] + [1]
            prods = [np.prod(prevs[i:]) for i in range(len(prevs))]
            ip_vars = ["cast<int>(floor({}/{}))%{}".format(dim_vars[1], prod, ips) for ips, prod in zip(
                ip.shape,
                prods)]
            print(ip_vars)
        else:
            op_shape = [np.prod(ip.shape[:axis])] + [np.prod(ip.shape[axis:])]
            pprevs = ip.shape[axis:] + [1]
            pprods = [np.prod(pprevs[i:]) for i in range(len(pprevs))]
            fprevs = ip.shape[:axis] + [1]
            fprods = [np.prod(fprevs[i:]) for i in range(len(fprevs))]
            ip_vars = ["cast<int>(floor({}/{}))%{}".format(dim_vars[0], prod, ips) for ips, prod in zip(
                fprevs,
                fprods[1:])] \
                      + ["cast<int>(floor({}/{}))%{}".format(dim_vars[1], prod, ips) for ips, prod in zip(
                          pprevs,
                          pprods[1:])]

        self.cpp("{}({}) = {}({});".format(
            op.name, ','.join(dim_vars[::-1]),
            ip.name, ','.join((ip_vars[::-1]))))

        op.set_shape(op_shape)
        op.set_type(ip.type)

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
        op_dim_vars = [dvar for i, dvar in enumerate(dim_vars)] \
                      if keepdims else \
                         [dvar for i, dvar in enumerate(dim_vars) \
                          if i != axis]
        ip_dim_vars = [(dvar if i != axis else "r") \
                       for i, dvar in enumerate(dim_vars)]
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

    def generate_concat(self, node):
        ip0 = self.funcs[node.input[0]]
        ip1 = self.funcs[node.input[1]]
        op  = self.funcs[node.output[0]]
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        n_dims = len(ip0.shape)
        op_shape = [ip0s + ip1s if i == axis else ip0s for i, (ip0s, ip1s) \
                    in enumerate(zip(ip0.shape, ip1.shape))]
        dim_vars = self.generate_dim_vars(n_dims)

        ip0_dim_vars = ["clamp({}, 0, {})".format(v,
                                                  ip0.shape[axis] - 1) \
                        if i == axis else v for i, v \
                        in enumerate(dim_vars)]
        ip1_dim_vars = ["clamp({} - {}, 0, {})".format(
            v,
            ip0.shape[axis],
            ip1.shape[axis] - 1) \
                        if i == axis else v for i, v \
                        in enumerate(dim_vars)]
        self.cpp("{}({}) = select({} < {}, {}({}), {}({}));".format(
            op.name, ','.join(dim_vars[::-1]),
            dim_vars[axis], ip0.shape[axis],
            ip0.name, ','.join(ip0_dim_vars[::-1]),
            ip1.name, ','.join(ip1_dim_vars[::-1])))

        op.set_shape(op_shape)
        op.set_type(ip0.type)

    def generate_gather(self, node):
        ip  = self.funcs[node.input[0]]
        ids = self.funcs[node.input[1]]
        op  = self.funcs[node.output[0]]
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i

        op_shape = ip.shape[:axis] + ids.shape + ip.shape[axis+1:]
        op.set_shape(op_shape)
        op.set_type(ip.type)
        dim_vars = self.generate_dim_vars(len(op_shape))
        self.cpp("{}({}) = {}({});".format(
            op.name, ','.join(dim_vars[::-1]),
            ip.name, ','.join((dim_vars[:axis] \
                               + ["clamp(cast<int>({}({})), 0, {})".format(ids.name, dim_vars[axis], ip.shape[axis]-1)] \
                               + dim_vars[axis+1:])[::-1])))

    def generate_gemm(self, node):
        A   = self.funcs[node.input[0]]
        B   = self.funcs[node.input[1]]
        C   = self.funcs[node.input[2]]
        Y   = self.funcs[node.output[0]]
        alpha, beta, transA, transB = 1, 1, 0, 0
        for attr in node.attribute:
            if attr.name == "alpha":
                alpha = attr.f
            if attr.name == "beta":
                beta = attr.f
            if attr.name == "transA":
                transA = attr.i
            if attr.name == "transB":
                transB = attr.i
        if transA:
            K, M = A.shape
        else:
            M, K = A.shape
        if transB:
            N, K = B.shape
        else:
            K, N = B.shape
        alpha = "cast<{}>(Expr({}))".format(C.type, alpha)
        beta = "cast<{}>(Expr({}))".format(C.type, beta)
        self.cpp("RDom r(0, {});".format(K))
        Y.set_shape([M, N])
        Y.set_type(C.type)
        dim_vars = self.generate_dim_vars(len(C.shape))
        self.generate_func("norm_A")
        self.generate_func("norm_B")
        self.generate_func("norm_C")
        if transA:
            self.cpp("norm_A({}) = {}({});".format(
                ','.join(dim_vars[:2][::-1]),
                A.name, ','.join(dim_vars[:2])))
        else:
            self.cpp("norm_A = {};".format(A.name))
        if transB:
            self.cpp("norm_B({}) = {}({});".format(
                ','.join(dim_vars[:2][::-1]),
                B.name, ','.join(dim_vars[:2])))
        else:
            self.cpp("norm_B = {};".format(B.name))
        self.cpp("norm_C({}) = {}({});".format(
            ','.join(dim_vars[::-1]),
            C.name, ','.join([dv if cs > 1 else "0" \
                              for dv, cs in zip(dim_vars[:2],
                                                C.shape)][::-1])))

        self.cpp("{}({}) = {}*norm_C({})+{}*sum(norm_A({})*norm_B({}));".format(
            Y.name, ','.join(dim_vars[::-1]),
            beta,
            ','.join(dim_vars[::-1]),
            alpha,
            ','.join([dim_vars[0], "r"][::-1]),
            ','.join(["r", dim_vars[1]][::-1])))

    def generate_hardmax(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        op.set_shape(ip.shape)
        op.set_type(ip.type)
        axis = 1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        dim_vars = self.generate_dim_vars(len(ip.shape))
        red_vars = self.generate_rdom(ip.shape[axis:])

        ip_vars = dim_vars[:axis] + red_vars
        self.cpp("Tuple am = argmax({}({}));".format(
            ip.name, JOIN_VARS(ip_vars)))
        self.cpp("{}({}) = cast<{}>({});".format(
            op.name, JOIN_VARS(dim_vars),
            op.type,
            "&&".join(["(am[{}]=={})".format(i, dv) for \
                       i, dv in enumerate(dim_vars[axis:])])))
            
    def generate_pad(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        const = "Expr(0)"
        pads = None
        mode = "constant"
        for attr in node.attribute:
            if attr.name == "mode":
                mode = attr.s.decode()
            if attr.name == "pads":
                pads = [(a, b) for a, b in zip(
                    attr.ints[:len(attr.ints)//2],
                    attr.ints[len(attr.ints)//2:])]
            if attr.name == "value":
                const = "cast<{}>(Expr({}))".format(
                    C_TYPE_DECODER(attr.type),
                    attr.f)
        dim_vars = self.generate_dim_vars(len(ip.shape))
        n_ign_dims = len(ip.shape) - len(pads)
        op_shape = [ips + pad[0] + pad[1] if pad else ips \
                    for pad, ips in zip(
                            [None] * n_ign_dims + pads,
                            ip.shape)]
        if mode == "constant":
            self.cpp("Func padded = BoundaryConditions::constant_exterior({}, {}, {{{}}});".format(
                ip.name,
                const, ','.join(["{{0, {}}}".format(ips) \
                                 for ips in ip.shape[::-1]])))
        elif mode == "edge":
            self.cpp("Func padded = BoundaryConditions::repeat_edge({}, {{{}}});".format(
                ip.name,
                ','.join(["{{0, {}}}".format(ips) \
                          for ips in ip.shape[::-1]])))
        elif mode == "reflect":
            self.cpp("Func padded = BoundaryConditions::miccor_image({}, {{{}}});".format(
                ip.name,
                ','.join(["{{0, {}}}".format(ips) \
                          for ips in ip.shape[::-1]])))
        ip_vars = ["{}-{}".format(dv, pad[0]) if pad else dv \
                   for dv, pad in zip(dim_vars,
                                      [None] * n_ign_dims + pads)]
        self.cpp("{}({}) = padded({});".format(
            op.name,
            ','.join(dim_vars[::-1]),
            ','.join(ip_vars[::-1])))

        op.set_shape(op_shape)
        op.set_type(ip.type)

    def generate_conv(self, node):
        ip    = self.funcs[node.input[0]]
        w     = self.funcs[node.input[1]]
        op    = self.funcs[node.output[0]]
        kernel_shape = None
        pads         = None
        padded       = False
        strides      = None
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
            if attr.name == "pads":
                li = len(attr.ints) // 2
                pads = [(a, b) for (a, b) in zip(attr.ints[:li],
                                                 attr.ints[li:])]
                padded = True
            if attr.name == "strides":
                strides = list(attr.ints)
        if not kernel_shape:
            kernel_shape = w.shape[2:]
        if not pads:
            pads = [(0, 0) for k in kernel_shape]
        if not strides:
            strides = [1 for i in w.shape[2:]]


        dim_vars = self.generate_dim_vars(len(ip.shape))
        self.cpp("RDom r({{{}}}, \"r\");".format(
            ','.join(["{{0,{}}}".format(i) \
                      for i in [ip.shape[1]] + kernel_shape])))
        red_vars= ["r[{}]".format(i) for i in range(len(kernel_shape) + 1)]

        ip_vars = [dim_vars[0]] + [red_vars[0]] + \
                  ["{}*{}+{}-{}".format(dv, stride, rv, pad[0]) for \
                   dv, rv, pad, stride in \
                   zip(dim_vars[2:],
                       red_vars[1:],
                       pads,
                       strides)]
        w_vars = [dim_vars[1]] + red_vars
        self.cpp()
        if padded:
            self.cpp("Func padded = BoundaryConditions::constant_exterior({}, 0, {{{}, {{Expr(), Expr()}}, {{Expr(), Expr()}}, }});".format(
                ip.name,
                ','.join(["{{0,{}}}".format(s) \
                          for i, s in enumerate(ip.shape[2:][::-1])])))
        else:
            self.cpp("Func padded = {};".format(ip.name))
        self.cpp("{}({}) = sum(padded({}) * {}({}));".format(
            op.name, ','.join(dim_vars[::-1]),
            ','.join(ip_vars[::-1]),
            w.name, ','.join(w_vars[::-1])))
        op_shape = [ip.shape[0], w.shape[0]] \
                   + [floor((ips+pad[0]+pad[1]-ks)/stride+1) \
                                   for (ips, pad, ks, stride) \
                                   in zip(ip.shape[2:],
                                          pads,
                                          w.shape[2:],
                                          strides)]
        op.set_shape(op_shape)
        op.set_type(ip.type)

    def generate_convT(self, node):
        ip    = self.funcs[node.input[0]]
        w     = self.funcs[node.input[1]]
        op    = self.funcs[node.output[0]]
        kernel_shape = None
        pads         = None
        strides      = None
        op_shape     = None
        op_pads      = None
        auto_pad     = None
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
            if attr.name == "pads":
                li = len(attr.ints) // 2
                pads = [(a, b) for (a, b) in zip(attr.ints[:li],
                                                 attr.ints[li:])]
            if attr.name == "strides":
                strides = list(attr.ints)
            if attr.name == "output_shape":
                op_shape = list(attr.ints)
            if attr.name == "output_padding":
                op_pads = list(attr.ints)
        if not strides:
            strides = [1 for i in w.shape[2:]]
        if not kernel_shape:
            kernel_shape = w.shape[2:]
        if not pads:
            pads = [(0, 0) for w in kernel_shape]
        if not op_pads:
            op_pads = [0 for j in kernel_shape]
        if not op_shape and op_pads:
            op_shape = [stride*(ips-1)+op_pad+ks-pad[0]-pad[1] \
                        for (ips, op_pad, ks, stride, pad) in
                        zip(ip.shape[2:],
                            op_pads,
                            kernel_shape,
                            strides,
                            pads)]
        if op_shape:
            op_shape = op_shape[-len(kernel_shape):]
            total_padding = [stride*(ops-1)+op_pad+ks-ips \
                             for (ips, ops, op_pad, ks, stride) in \
                             zip(ip.shape[2:],
                                 op_shape,
                                 op_pads,
                                 kernel_shape,
                                 strides)]
            if auto_pad:
                if auto_pad != "SAME_UPPER":
                    pads = [(tp//2, tp-tp//2) for tp in total_padding]
                else:
                    pads = [(tp-tp//2, tp//2) for tp in total_padding]
        if not op_shape:
            op_shape = [ip.shape[0], w.shape[1]] \
                   + [stride*(ips-1)+op_pad+ks-pad[0]-pad[1]#floor((ips+pad[0]+pad[1]-ks)/stride+1) \
                      for (ips, pad, ks, stride, op_pad) \
                      in zip(ip.shape[2:],
                             pads,
                             w.shape[2:],
                             strides,
                             op_pads
                      )]
        op_shape = [ip.shape[0], w.shape[1]] + op_shape
        dim_vars = self.generate_dim_vars(len(ip.shape))
        self.cpp("RDom r({{{}}}, \"r\");".format(
            ','.join(["{{0,{}}}".format(i) \
                      for i in ip.shape[1:]])))
        red_vars= ["r[{}]".format(i) for i in range(len(kernel_shape) + 1)]

        ip_vars = [dim_vars[0]] + red_vars
        w_vars = [dim_vars[0]] + [red_vars[0]] + \
                 ["-{}*{}+{}+{}".format(rv, stride, dv, pad[0]) for \
                  dv, rv, pad, stride in \
                  zip(dim_vars[2:],
                      red_vars[1:],
                      pads,
                      strides)]
        self.cpp()
        self.cpp("Func padded = BoundaryConditions::constant_exterior({}, 0, {{{}, {{Expr(), Expr()}}, {{Expr(), Expr()}}, }});".format(
            w.name,
            ','.join(["{{0,{}}}".format(s) \
                      for i, s in enumerate(w.shape[2:][::-1])])))

        self.cpp()
        self.cpp("{}({}) = sum({}({}) * padded({}));".format(
            op.name, ','.join(dim_vars[::-1]),
            ip.name, ','.join(ip_vars[::-1]),
            ','.join(w_vars[::-1])))

        op.set_shape(op_shape)
        op.set_type(ip.type)

    def generate_dtos(self, node):
        ip  = self.funcs[node.input[0]]
        op  = self.funcs[node.output[0]]
        N, C, H, W = ip.shape
        blocksize = None
        for attr in node.attribute:
            if attr.name == "blocksize":
                blocksize = attr.i
        op.set_shape([N, C//(blocksize*blocksize), H*blocksize, W*blocksize])
        op.set_type(ip.type)
        dim_vars = self.generate_dim_vars(4)

        ip_vars = [dim_vars[0],
                   "{}+({}%{})*{}+({}%{})*{}".format(
                       dim_vars[1],
                       dim_vars[3], blocksize, ip.shape[1] // (blocksize**2),
                       dim_vars[2], blocksize, ip.shape[1] // blocksize),
                   "cast<int>({}/{})".format(
                       dim_vars[2], blocksize),
                   "cast<int>({}/{})".format(
                       dim_vars[3],
                       blocksize)]
        self.cpp("{}({}) = {}({});".format(
            op.name, ','.join(dim_vars[::-1]),
            ip.name, ','.join(ip_vars[::-1])
        ))
    def generate_pool(self, node):
        ip    = self.funcs[node.input[0]]
        op    = self.funcs[node.output[0]]

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
                li = len(attr.ints)//2
                pads = [(a, b) for a, b in zip(attr.ints[:li],
                                               attr.ints[li:])]
                padded = sum(attr.ints) > 0
            if attr.name == "auto_pad":
                auto_pad = attr.s.decode()
            if attr.name == "strides":
                strides = list(attr.ints)
        if not pool_shape:
            pool_shape = ip.shape[2:]
        if not pads:
            if auto_pad == "SAME_UPPER":
                pads = [(floor((ks-1)/2), ceil((ks-1)/2)) \
                        for ks in pool_shape]
                padded = True
            elif auto_pad == "SAME_LOWER":
                pads = [(ceil((ks-1)/2), floor((ks-1)/2)) \
                        for ks in pool_shape]
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
        ip_vars = ["{}*{}+{}-{}".format(dv, st, rv, pad[0]) \
                   if rv else dv \
                   for (dv, rv, st, pad) in zip(
                           dim_vars,
                           [None] * n_ign_dims + red_vars,
                           [None] * n_ign_dims + strides,
                           [None] * n_ign_dims + pads)]

        if padded:
            self.cpp("Func padded = BoundaryConditions::constant_exterior({}, 0, {{{}}});".format(
                ip.name,
                ','.join(["{{0,{}}}".format(s) \
                          if i < len(red_vars) else "{Expr(),Expr()}" \
                          for i, s in enumerate(ip.shape[::-1])])))
        else:
            self.cpp("Func padded = {};".format(ip.name))
        self.cpp()
        self.cpp("Func counts;")

        if node.op_type in ["AveragePool", "GlobalAveragePool"]:
            if count_include_pad:
                self.cpp("counts({}) = {};".format(
                    ','.join(dim_vars[::-1][:len(red_vars)]),
                    filter_area))
            else:
                self.cpp("Func ones;")
                self.cpp("ones({}) = 1;".format(
                    ','.join(dim_vars[::-1][:len(red_vars)])))
                self.cpp("Func padded_ones = BoundaryConditions::constant_exterior(ones, 0, {{{}}});".format(
                    ','.join(["{{0,{}}}".format(s) \
                              for s in ip.shape[::-1][:len(red_vars)]])))
                self.cpp("counts({}) = sum(padded_ones({}));".format(
                    ','.join(dim_vars[::-1][:len(red_vars)]),
                    ','.join(ip_vars[::-1][:len(red_vars)])
                ))
            self.cpp("{}({}) = sum(padded({})) / counts({});".format(
                op.name, ','.join(dim_vars[::-1]),
                ','.join(ip_vars[::-1]),
                ','.join(dim_vars[::-1][:len(red_vars)])))
        elif node.op_type in ["MaxPool", "GlobalMaxPool"]:
            self.cpp("{}({}) = maximum(padded({}));".format(
                op.name, ','.join(dim_vars[::-1]),
                ','.join(ip_vars[::-1])))

        op_shape = [floor((ips+pad[0]+pad[1]-ks)/stride+1) \
                    if pad else ips \
                    for (ips, pad, ks, stride) \
                    in zip(ip.shape,
                           [None] * n_ign_dims + pads,
                           [None] * n_ign_dims + pool_shape,
                           [None] * n_ign_dims + strides)]

        op.set_shape(op_shape)
        op.set_type(ip.type)
        return

    def generate_node(self, nidx, node):
        if node.op_type == "Constant":
            return
        for op in node.output:
            f_name = op.replace('/', '') + "_func"
            self.generate_func(f_name)
            assert(op not in self.funcs)
            self.funcs[op] = HalideObj(f_name,)
        self.cpp("{{ // {} {} {}".format(node.op_type, nidx, node.name), 1)
        if   node.op_type == "Abs":
            self.generate_unary_expr(node, "abs({})")
        elif node.op_type == "Acos":
            self.generate_unary_expr(node, "acos({})")
        elif node.op_type == "Asin":
            self.generate_unary_expr(node, "asin({})")
        elif node.op_type == "Atan":
            self.generate_unary_expr(node, "atan({})")
        elif node.op_type == "Ceil":
            self.generate_unary_expr(node, "ceil({})")
        elif node.op_type == "Cos":
            self.generate_unary_expr(node, "cos({})")
        elif node.op_type == "Clip":
            self.generate_unary_expr(node, "clamp({}, {}, {})")
        elif node.op_type == "Cast":
            self.generate_unary_expr(node, "cast<{1}>({0})")
        elif node.op_type == "Dropout":
            self.generate_unary_expr(node, "{}")
        elif node.op_type == "Elu":
            self.generate_unary_expr(node, "select({0} < 0, cast<{2}>(Expr({1}) * (exp({0}) - Expr(1.))), {0})")
        elif node.op_type == "Exp":
            self.generate_unary_expr(node, "exp({})")
        elif node.op_type == "Floor":
            self.generate_unary_expr(node, "floor({})")
        elif node.op_type == "Add":
            self.generate_bin_expr(node, "+")
        elif node.op_type == "And":
            self.generate_bin_expr(node, "&")
        elif node.op_type == "Div":
            self.generate_bin_expr(node, "/")
        elif node.op_type == "Equal":
            self.generate_bin_expr(node, "==", "int8_t")
        elif node.op_type == "Greater":
            self.generate_bin_expr(node, ">", "int8_t")
        elif node.op_type == "ArgMax":
            self.generate_red_expr(node, "argmax", "int64_t")
        elif node.op_type == "ArgMin":
            self.generate_red_expr(node, "argmin", "int64_t")
        elif node.op_type == "BatchNormalization":
            self.generate_bn(node)
        elif node.op_type == "AveragePool":
            self.generate_pool(node)
        elif node.op_type == "GlobalAveragePool":
            self.generate_pool(node)
        elif node.op_type == "GlobalMaxPool":
            self.generate_pool(node)
        elif node.op_type == "Conv":
            self.generate_conv(node)
        elif node.op_type == "ConvTranspose":
            self.generate_convT(node)
        elif node.op_type == "Concat":
            self.generate_concat(node)
        elif node.op_type == "Pad":
            self.generate_pad(node)
        elif node.op_type == "DepthToSpace":
            self.generate_dtos(node)
        elif node.op_type == "Flatten":
            self.generate_flatten(node)
        elif node.op_type == "Gather":
            self.generate_gather(node)
        elif node.op_type == "Gemm":
            self.generate_gemm(node)
        elif node.op_type == "Hardmax":
            self.generate_hardmax(node)
        elif node.op_type == "GRU":
            raise NotImplementedError
        elif node.op_type == "Expand":
            raise NotImplementedError
        else:
            print()
            print("unhandled node ", node.op_type)
            print(node)
            raise NotImplementedError
        self.cpp("}", -1)

        for op in node.output:
            try:
                self.cpp("// {} {}{}".format(op, self.funcs[op].type,
                                              self.funcs[op].shape))
                self.funcs[op].type
            except:
                print(node)
                print(self.halide_str)

        #self.cpp("{}.realize();".format(node.output[0]))
