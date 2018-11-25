from onnx.backend.base import BackendRep
from onnx import TensorProto, shape_inference
import subprocess
import ctypes
import _ctypes
import numpy as np
import importlib
import os
from math import floor, ceil
from .tensortypes import HalogenType


GLOBAL_LIDX = 1 # Hack to avoid library naming collisions
if "HALIDE_DIR" in os.environ:
    HALIDE_DIR = os.environ['HALIDE_DIR']
else:
    HALIDE_DIR = "/usr/local"

JOIN_VARS = lambda vars: ','.join(vars[::-1])

CAST = lambda expr, type: "cast<{}>(Expr({}))".format(type, expr)
class OnnxAttr:
    def __init__(self, attr, v_fn=lambda x:x, value=None, type="NOTYPE"):
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
    def __init__(self, name=None, shape=-1, type=None, io=0):
        self._name = name
        self._shape = -1 if shape == -1 else \
                      tuple([int(i) for i in shape])
        self._type = type
        self._io = io
    @property
    def name(self):
        assert(self._name)
        return self._name
    @property
    def shape(self):
        assert(self._shape != -1)
        return list(self._shape)
    @property
    def size(self):
        return int(np.prod(self.shape))
    @property
    def is_scalar(self):
        assert(self._shape != -1)
        return self._shape == ()
    @property
    def type(self):
        assert(self._type)
        return self._type
    @property
    def is_input(self):
        assert(self._io != 0)
        return self._io == 1
    @property
    def is_output(self):
        assert(self._io != 0)
        return self._io == -1
    def set_shape(self, shape):
        assert(all([type(i) == int for i in shape]))
        assert(self._shape == -1 or self._shape == tuple(shape))
        self._shape = tuple(shape)
    def set_type(self, typ):
        assert(not self._type or self._type == typ)
        assert(type(typ) == HalogenType)
        self._type = typ
    def __repr__(self):
        return "({}, {}, {}, {})".format(self._name,
                                         self._shape,
                                         self._type,
                                         {1:"INPUT",
                                          0:"NODE",
                                          -1:"OUTPUT"}[self._io])

def is_loaded(lib):
    libp = os.path.abspath(lib)
    ret = os.system("lsof -w -p %d | grep %s > /dev/null" % (os.getpid(), libp))
    return ret == 0

class HalideBackendRep(BackendRep):
    def __init__(self, model):
        global GLOBAL_LIDX
        self.halogen_str = """"""
        self.indent = 0
        self.name_map = {}
        self.model_name = "{}_{}_{}_{}".format(model.graph.name,
                                               model.model_version,
                                               model.domain.replace('.', '-'),
                                               GLOBAL_LIDX)
        GLOBAL_LIDX += 1
        self.generate_csrc(model)
    def __del__(self):
        try:
            os.remove("generated/lib{}.so".format(self.model_name))
        except FileNotFoundError:
            pass
    
    def cpp(self, s="", incr=0):
        if incr < 0:
            self.indent += incr
        if s:
            self.halogen_str += "  " * self.indent
        self.halogen_str += "{}\n".format(s)
        if incr > 0:
            self.indent += incr

    def run(self, inputs, **kwargs):
        i = 0
        args = []
        outputs = []
        for name, ctype in zip(self.c_args, self.halide_fn.argtypes):
            func = self.funcs[name]
            if func.is_input:
                if i >= len(inputs):
                    input = self.init_data[name]
                else:
                    input = inputs[i]
                    assert(tuple(input.shape) == tuple(func.shape))
                if func.is_scalar:
                    args.append(ctype(input))
                else:
                    args.append(input.ctypes.data_as(ctype))
                i += 1
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
        self.cpp("#include \"Halide.h\"")
        self.cpp("#include <stdint.h>")
        self.cpp("#include <cfloat>")
        self.cpp("#include <limits.h>")
        self.cpp()

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

        self.cpp("using namespace Halide;")
        self.cpp("namespace {")

        self.cpp("class HalOGen : public Generator<HalOGen> {", 1)
        self.cpp("public:", 1)
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
                self.cpp("{2}<{0}> {1}{{\"{1}\"}};".format(
                    type.c,
                    onnx_name + ("" if is_output else "_s"),
                    "Output" if is_output else "Input "))
                if not is_output:
                    input_scalars.append(self.funcs[tensor.name])
            else:
                self.cpp("{4}<Buffer<{0}>> {1}{{\"{1}\", {2}}}; //{0}{3}".format(
                    type.c,
                    onnx_name,
                    len(c_shape),
                    c_shape,
                    "Output" if is_output else "Input "))

            self.c_args.append(tensor.name)

        # Generate the Halide compute function
        self.cpp()
        self.cpp("void generate() {", 1);
        for ip in input_scalars:
            self.generate_func(ip.name)
            self.cpp("{}() = {}_s;".format(
                ip.name, ip.name))
        # Generate Funcs for operator nodes
        for nidx, node in enumerate(model.graph.node):
            self.cpp()
            self.generate_node(nidx, node)

        self.cpp("};", -1)
        self.cpp("", -1)
        self.cpp("};")
        self.cpp("}", -1)
        self.cpp("HALIDE_REGISTER_GENERATOR(HalOGen, halogen)")


        with open("generated/halogen_generator.cpp", 'w') as f:
            f.write(self.halogen_str)

        # Generate C shim to Halide generated code
        self.halogen_str = """"""
        self.cpp("#include \"Halide.h\"")
        self.cpp("#include \"halogen.h\"")
        self.cpp("using float16_t = Halide::float16_t;")
        self.cpp("using namespace Halide::Runtime;")
        self.cpp("extern \"C\" {", 1)
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
        self.cpp("int halogen_c({}) {{".format(','.join(py_args)), 1)
        for buf in buffers:
            self.cpp(buf)
        self.cpp("int r = halogen({});".format(','.join(ha_args)))
        for op in output_s:
            self.cpp(op)
        self.cpp("return r;")
        self.cpp("}", -1)
        self.cpp("}", -1)
        with open("generated/halogen_c.cpp", 'w') as f:
            f.write(self.halogen_str)

        cmd  = "g++ -std=c++11 -I {0}/include/ -I {0}/tools/ -g -fno-rtti "
        cmd += "generated/halogen_generator.cpp {0}/tools/GenGen.cpp {0}/lib/libHalide.a "
        cmd += "-o generated/halogen.generator -ldl -lpthread -lz -lrt -ldl -ltinfo -lm"
        cmd = cmd.format(HALIDE_DIR)
        r = subprocess.run(cmd, check=True, shell=True)

        cmd  = "generated/halogen.generator -g halogen -o generated -e "
        cmd += "assembly,bitcode,h,html,o,static_library,stmt,schedule "
        cmd += "target=host"
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
 

    def generate_var(self, var):
        self.cpp("Var {0};".format(var))

    def generate_func(self, fname):
        self.cpp("Func {0};".format(fname))

    def generate_dim_vars(self, n_vars):
        dim_vars = ["d_{}".format(i) for i in range(n_vars)]
        for dim_var in dim_vars:
            self.generate_var(dim_var)
        return dim_vars

    def generate_rdom(self, shape):
        self.cpp("RDom r({});".format(','.join(["0,{}".format(s) \
                                                for s in shape])))
        return ["r[{}]".format(i) for i in range(len(shape))]

    def generate_shape(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        op.set_shape([len(ip.shape)])
        op.set_type(HalogenType.from_c("int64_t"))
        self.cpp("int64_t* shape_c = new int64_t[{0}] {{{1}}};".format(
            len(ip.shape),
            ','.join(map(str, ip.shape))))
        self.cpp("Buffer<int64_t> shape_buf(shape_c, {{{}}});".format(len(ip.shape)))
        self.cpp("Func shape_func(shape_buf);")
        dim_var = self.generate_dim_vars(1)[0]
        self.cpp("{}({}) = shape_func({});".format(
            op.name, dim_var, dim_var))

    def generate_size(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        op.set_shape(())
        op.set_type(HalogenType.from_c("int64_t"))
        self.cpp("{}() = cast<int64_t>(Expr({}));".format(
            op.name, ip.size))
            
    def generate_constant(self, node):
        op = self.funcs[node.output[0]]
        tensor    = node.attribute[0].t
        op.set_shape(tuple(tensor.dims))
        op.set_type(HalogenType.from_onnx(tensor.data_type))

        if tensor.raw_data:
            const_data = np.frombuffer(
                tensor.raw_data,
                count=op.size,
                dtype=op.type.np)
        else:
            const_data = tensor.float_data
        if op.is_scalar:
            self.cpp("{}() = Expr({});".format(op.name, const_data[0]))
        else:
            self.cpp("{0}* const_a = new {0}[{1}] {{{2}}};".format(
                op.type.c,
                op.size,
                ','.join(map(str,const_data))))
            self.cpp("Buffer<{}> const_b(const_a, {{{}}});".format(
                op.type.c,
                JOIN_VARS([str(i) for i in op.shape])))
            dim_vars = self.generate_dim_vars(len(op.shape))
            self.cpp("{}({}) = const_b({});".format(
                op.name,
                JOIN_VARS(dim_vars),
                JOIN_VARS(dim_vars),
                JOIN_VARS(dim_vars)))
    def generate_unary_expr(self, node, expr):
        ip      = self.funcs[node.input[0]]
        op      = self.funcs[node.output[0]]
        op1     = None
        if len(node.output) > 1:
            op1 = self.funcs[node.output[1]]
        op_type = ip.type
        min_v   = op_type.c_min
        max_v   = op_type.c_max
        alpha   = None
        if node.op_type in ["Elu", "ThresholdedRelu"]:
            alpha = 1.0
        elif node.op_type == "HardSigmoid":
            alpha = 0.2
        elif node.op_type == "LeakyRelu":
            alpha = 0.01
        elif node.op_type == "Selu":
            alpha = 1.67326
        beta    = 0.5
        gamma   = 1.0507
        for attr in node.attribute:
            if attr.name == "alpha":
                alpha = attr.f
            if attr.name == "beta":
                beta = attr.f
            if attr.name == "gamma":
                gamma = attr.f
            if attr.name == "to":
                op_type = HalogenType.from_onnx(attr.i)
            if attr.name == "max":
                max_v = "Expr({})".format(attr.f)
            if attr.name == "min":
                min_v = "Expr({})".format(attr.f)

        alpha = "cast<{}>(Expr({}))".format(op_type.c, alpha)
        beta  = "cast<{}>(Expr({}))".format(op_type.c, beta)
        gamma = "cast<{}>(Expr({}))".format(op_type.c, gamma)
        dim_vars = self.generate_dim_vars(len(ip.shape))

        ip_expr = "{}({})".format(ip.name, ','.join(dim_vars[::-1]))

        if node.op_type == "Cast":
            expr = expr.format(ip_expr, op_type.c)
        elif node.op_type == "Clip":
            expr = expr.format(ip_expr, min_v, max_v)
        elif node.op_type == "Elu":
            expr = expr.format(ip_expr, alpha, ip.type.c)
        elif node.op_type == "HardSigmoid":
            expr = expr.format(ip_expr, alpha, beta)
        elif node.op_type == "LeakyRelu":
            expr = expr.format(ip_expr, alpha)
        elif node.op_type == "Selu":
            expr = expr.format(ip_expr, alpha, gamma)
        elif node.op_type == "ThresholdedRelu":
            expr = expr.format(ip_expr, alpha)
        else:
            expr = expr.format(ip_expr)


        self.cpp("{}({}) = {};".format(
            op.name, ','.join(dim_vars[::-1]),
            expr))
        if op1:
            self.cpp("{}({}) = Expr(1.);".format(
                op1.name, JOIN_VARS(dim_vars)))
            op1.set_shape(ip.shape)
            op1.set_type(ip.type)
        op.set_shape(ip.shape)
        op.set_type(op_type)

    def generate_var_expr(self, node):
        ips = [self.funcs[i] for i in node.input]
        op  = self.funcs[node.output[0]]
        dims = max([len(ip.shape) for ip in ips])
        dim_vars = self.generate_dim_vars(dims)
        op_shape = [1] * dims
        ip_vars = []
        op.set_type(ips[0].type)
        if (len(ips) == 1):
            op.set_shape(ips[0].shape)
            self.cpp("{}({}) = {}({});".format(
                op.name, JOIN_VARS(dim_vars),
                ips[0].name, JOIN_VARS(dim_vars)))
        else:
            for ip in ips:
                op_shape = [op_shape[-i] if i > len(ip.shape) else
                            max(ip.shape[-i], op_shape[-i]) for \
                            i in range(1, dims+1)][::-1]
                ip_vars.append([(dv if dim > 1 else "0") for dim, dv in
                                zip(ip.shape, dim_vars[-len(ip.shape):])])
            op.set_shape(op_shape)

            if node.op_type in ["Max", "Min"]:
                self.cpp("{}({}) = {}({});".format(
                    op.name, JOIN_VARS(dim_vars),
                    node.op_type.lower(),
                    JOIN_VARS(["{}({})".format(ip.name, JOIN_VARS(ipv)) for
                               ip, ipv in
                               zip(ips, ip_vars)])))
            elif node.op_type == "Sum":
                self.cpp("{}({}) = {};".format(
                    op.name, JOIN_VARS(dim_vars),
                    '+'.join(["{}({})".format(ip.name, JOIN_VARS(ipv)) for
                             ip, ipv in
                             zip(ips, ip_vars)])))
            else:
                self.cpp("{}({}) = ({})/cast<{}>(Expr({}));".format(
                    op.name, JOIN_VARS(dim_vars),
                    '+'.join(["{}({})".format(ip.name, JOIN_VARS(ipv)) for
                               ip, ipv in
                               zip(ips, ip_vars)]),
                    op.type.c, len(ips)))

    def generate_prelu_expr(self, node):
        ip0 = self.funcs[node.input[0]]
        ip1 = self.funcs[node.input[1]]
        op  = self.funcs[node.output[0]]
        op.set_type(ip0.type)
        op.set_shape(ip0.shape)
        dim_vars = self.generate_dim_vars(len(ip0.shape))
        ip1_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in \
                        zip(ip1.shape[::-1], dim_vars[::-1])][::-1]
        ip0_expr = "{}({})".format(ip0.name,
                                   JOIN_VARS(dim_vars))
        ip1_expr = "{}({})".format(ip1.name,
                                   JOIN_VARS(ip1_dim_vars))
        self.cpp("{0}({1}) = select({2}<0, {2}*{3}, {2});".format(
            op.name, JOIN_VARS(dim_vars),
            ip0_expr, ip1_expr))
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
        ip0_expr = "{}({})".format(ip0.name, JOIN_VARS(ip0_dim_vars))

        ip1_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(ip1_dim, dim_vars[-len(ip1_dim):])]
        ip1_expr = "{}({})".format(ip1.name, JOIN_VARS(ip1_dim_vars))


        expr = expr.format(ip0_expr, ip1_expr)

        self.cpp("{}({}) = cast<{}>({});".format(
            op.name, ",".join(dim_vars[::-1]),
            op_type.c,
            expr,
            ))

        op.set_shape(
            [ip1_dim[-i] if i > len(ip0_dim) else
             (ip0_dim[-i] if i > len(ip1_dim) else
              max(ip0_dim[-i], ip1_dim[-i])) \
             for i in range(1, dims+1)][::-1])
        op.set_type(op_type)

    def generate_reshape(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        shape = node.input[1]
        if shape not in self.init_data:
            print("Reshapes must be known at compile time")
            with open("generated/halogen_generator.cpp", "w") as f:
                f.write(self.halogen_str)
            exit(1)
        shape = tuple([int(i) for i in self.init_data[shape]])
        op.set_shape(shape)
        op.set_type(ip.type)
        dim_vars = self.generate_dim_vars(len(op.shape))

        prevs = ip.shape[1:] + [1]
        prods = [np.prod(prevs[i:]) for i in range(len(prevs))]
        ip_vars = ["cast<int>(floor({}/{}))%{}".format(dim_vars[0],
                                                       prod,
                                                       ips) for ips, prod in \
                   zip(ip.shape, prods)]
        
        self.generate_func("flattened")
        self.cpp("flattened({}) = {}({});".format(
            dim_vars[0],
            ip.name, JOIN_VARS(ip_vars)))
        prevs = op.shape[1:] + [1]
        prods = [np.prod(prevs[i:]) for i in range(len(prevs))]
        fl_vars = ["({}*{})".format(dv, p) for dv, p in zip(dim_vars, prods)]
        self.cpp("{}({}) = flattened({});".format(
            op.name, JOIN_VARS(dim_vars),
            '+'.join(fl_vars[::-1])))
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
            op_shape = [1, ip.size]
            prevs = ip.shape[1:] + [1]
            prods = [np.prod(prevs[i:]) for i in range(len(prevs))]
            ip_vars = ["cast<int>(floor({}/{}))%{}".format(dim_vars[1], prod, ips) for ips, prod in zip(
                ip.shape,
                prods)]
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

        op.set_shape([int(i) for i in op_shape])
        op.set_type(ip.type)

    def generate_redl_expr(self, node, expr):
        ip    = self.funcs[node.input[0]]
        op    = self.funcs[node.output[0]]
        keepdims = True
        axes = [i for i in range(len(ip.shape))]
        for attr in node.attribute:
            if attr.name == "keepdims":
                keepdims = attr.i == 1
            if attr.name == "axes":
                axes = list(attr.ints)

        if keepdims:
            op_shape = [s if i not in axes else 1 for i, s in enumerate(ip.shape)]
        else:
            op_shape = [s for i, s in enumerate(ip.shape) if i not in axes]
        op.set_shape(op_shape)
        op.set_type(ip.type)
        red_shape = np.prod([ip.shape[i] for i in axes])
        red_vars = list(zip(axes, self.generate_rdom([ip.shape[i] for i in axes])))
        dim_vars = self.generate_dim_vars(len(op.shape))
        if keepdims:
            zd_vars = list(zip([i for i in range(len(ip.shape)) if i not in axes], [dv for i, dv in enumerate(dim_vars) if i not in axes]))
        else:
            zd_vars = list(zip([i for i in range(len(ip.shape)) if i not in axes], dim_vars))
        ip_vars = [None] * len(ip.shape)
        for i, rv in red_vars:
            ip_vars[i] = rv
        for i, dv in zd_vars:
            ip_vars[i] = dv

        if node.op_type == "ReduceMean":
            expr = expr.format("{}({})".format(ip.name, JOIN_VARS(ip_vars)),
                               red_shape)
        else:
            expr = expr.format("{}({})".format(ip.name, JOIN_VARS(ip_vars)))
        self.cpp("{}({}) = {};".format(
            op.name, JOIN_VARS(dim_vars),
            expr))


    def generate_red_expr(self, node, expr, type=None):
        type  = HalogenType.from_c(type)
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
            op_type.c,
            expr,
            ip.name, ','.join(ip_dim_vars[::-1])
            ))

        op.set_shape(op_shape)
        op.set_type(op_type)

    def generate_in(self, node):
        x    = self.funcs[node.input[0]]
        s    = self.funcs[node.input[1]]
        b    = self.funcs[node.input[2]]
        op   = self.funcs[node.output[0]]
        eps  = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = attr.f
        op.set_shape(x.shape)
        op.set_type(x.type)
        eps = "cast<{}>(Expr({}))".format(op.type.c, eps)
        red_vars = self.generate_rdom(op.shape[2:])
        dim_vars = self.generate_dim_vars(len(op.shape))
        self.generate_func("mean_f")
        self.generate_func("var_f")
        self.cpp("mean_f({}) = sum({}({}))/{};".format(
            JOIN_VARS(dim_vars[:2]),
            x.name, JOIN_VARS(dim_vars[:2] + red_vars),
            np.prod(x.shape[2:])))
        self.cpp("var_f({}) = sum(pow({}({}), 2))/{} - pow(mean_f({}), 2);".format(
            JOIN_VARS(dim_vars[:2]),
            x.name, JOIN_VARS(dim_vars[:2] + red_vars),
            np.prod(x.shape[2:]),
            JOIN_VARS(dim_vars[:2])))
        self.cpp("{}({}) = {}({})*({}({})-mean_f({}))/(sqrt(var_f({}) + {})) + {}({});".format(
            op.name, JOIN_VARS(dim_vars),
            s.name, dim_vars[1],
            x.name, JOIN_VARS(dim_vars),
            JOIN_VARS(dim_vars[:2]),
            JOIN_VARS(dim_vars[:2]),
            eps,
            b.name, dim_vars[1]))
    def generate_bn(self, node):
        x    = self.funcs[node.input[0]]
        s    = self.funcs[node.input[1]]
        bias = self.funcs[node.input[2]]
        mean = self.funcs[node.input[3]]
        var  = self.funcs[node.input[4]]
        op   = self.funcs[node.output[0]]
        eps  = 0
        eps_t = HalogenType.from_c("float")
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps   = attr.f
                eps_t = HalogenType.from_onnx(attr.type)
        #s * (x - mean) / np.sqrt(var + epsilon) + bias
        dim_vars = self.generate_dim_vars(len(x.shape))
        self.cpp("Expr eps({});".format(eps))
        self.cpp("{}({}) = cast<{}>({}({})*({}({}) - {}({})) / sqrt({}({})+eps) + {}({}));".format(
            op.name, ','.join(dim_vars[::-1]),
            x.type.c,
            s.name, dim_vars[1],
            x.name, ','.join(dim_vars[::-1]),
            mean.name, dim_vars[1],
            var.name, dim_vars[1],
            bias.name, dim_vars[1]))
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
        id_vars = dim_vars[axis:axis+len(ids.shape)]
        ip_vars = dim_vars[:axis] \
                  + ["clamp(cast<int>({}({})), 0, {})".format(ids.name,
                                                              JOIN_VARS(id_vars),
                                                              ip.shape[axis]-1)] \
                  + dim_vars[len(ids.shape)+axis:]
        self.cpp("{}({}) = {}({});".format(
            op.name, JOIN_VARS(dim_vars),
            ip.name, JOIN_VARS(ip_vars)))
    def generate_split(self, node):
        ip  = self.funcs[node.input[0]]
        ops = [self.funcs[o] for o in node.output]
        axis = 0
        split = None
        for attr in node.attribute:
            if attr.name == "split":
                split = list(attr.ints)
            if attr.name == "axis":
                axis = attr.i
        if not split:
            split = [int(ip.shape[axis] // len(ops))] * len(ops)
        dim_vars = self.generate_dim_vars(len(ip.shape))
        s_sum = 0
        for op, s in zip(ops, split):
            op.set_type(ip.type)
            op_shape = ip.shape
            op_shape[axis] = s
            op.set_shape(op_shape)
            ip_vars = list(dim_vars)
            ip_vars[axis] += "+{}".format(s_sum)
            self.cpp("{}({}) = {}({});".format(
                op.name, JOIN_VARS(dim_vars),
                ip.name, JOIN_VARS(ip_vars)))
            s_sum += s

    def generate_transpose(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        perms = list(range(len(ip.shape)))[::-1]
        for attr in node.attribute:
            if attr.name == "perm":
                perms = list(attr.ints)

        op.set_shape([ip.shape[i] for i in perms])
        op.set_type(ip.type)

        dim_vars = self.generate_dim_vars(len(ip.shape))
        self.cpp("{}({}) = {}({});".format(
            op.name, JOIN_VARS([dim_vars[i] for i in perms]),
            ip.name, JOIN_VARS(dim_vars)))
    def generate_unsqueeze(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        axes = None
        for attr in node.attribute:
            if attr.name == "axes":
                axes = list(attr.ints)
        op_shape = ip.shape
        orig_s = [1] * len(ip.shape)
        for i in axes:
            op_shape.insert(i, 1)
            orig_s.insert(i, 0)
        dim_vars = self.generate_dim_vars(len(op_shape))
        ip_vars = [dv for i, dv in enumerate(dim_vars) if orig_s[i]]
        op.set_shape(op_shape)
        op.set_type(ip.type)
        self.cpp("{}({}) = {}({});".format(
            op.name, JOIN_VARS(dim_vars),
            ip.name, JOIN_VARS(ip_vars)))
    def generate_squeeze(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        axes = None
        for attr in node.attribute:
            if attr.name == "axes":
                axes = list(attr.ints)

        op_shape = [s for i, s in enumerate(ip.shape) if i not in axes]
        op.set_shape(op_shape)
        op.set_type(ip.type)
        dim_vars = self.generate_dim_vars(len(op_shape))
        ip_vars = ["0"] * len(ip.shape)
        for i, dv in zip([i for i in range(len(ip.shape)) if i not in axes],
                         dim_vars):
            ip_vars[i] = dv
        self.cpp("{}({}) = {}({});".format(
            op.name, JOIN_VARS(dim_vars),
            ip.name, JOIN_VARS(ip_vars)))
    def generate_matmul(self, node):
        a   = self.funcs[node.input[0]]
        b   = self.funcs[node.input[1]]
        c   = self.funcs[node.output[0]]

        K = a.shape[-1]
        red_var = self.generate_rdom([K])[0]
        c.set_type(a.type)
        # TODO : Add broadcasting here
        if len(a.shape) == len(b.shape) == 2:
            c.set_shape([a.shape[0], b.shape[1]])
            dim_vars = self.generate_dim_vars(len(c.shape))
            a_vars = [dim_vars[0], red_var]
            b_vars = [red_var, dim_vars[1]]
        elif len(a.shape) > 2 and len(b.shape) == 2:
            c.set_shape(a.shape[:-1] + [b.shape[1]])
            dim_vars = self.generate_dim_vars(len(c.shape))
            a_vars = dim_vars[:-1] + [red_var]
            b_vars = [red_var, dim_vars[-1]]
        elif len(b.shape) > 2 and len(a.shape) == 2:
            c.set_shape(b.shape[:-2] + [a.shape[0], b.shape[-1]])
            dim_vars = self.generate_dim_vars(len(c.shape))
            a_vars = [dim_vars[-2], red_var]
            b_vars = dim_vars[:-2] + [red_var, dim_vars[-1]]
        elif len(a.shape) > 2 and len(b.shape) > 2:
            c.set_shape(b.shape[:-2] + [a.shape[-2], b.shape[-1]])
            dim_vars = self.generate_dim_vars(len(c.shape))
            a_vars = dim_vars[:-1] + [red_var]
            b_vars = dim_vars[:-2] + [red_var, dim_vars[-1]]


        self.cpp("{}({}) = sum({}({})*{}({}));".format(
            c.name, JOIN_VARS(dim_vars),
            a.name, JOIN_VARS(a_vars),
            b.name, JOIN_VARS(b_vars)))
            
        
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
        alpha = "cast<{}>(Expr({}))".format(C.type.c, alpha)
        beta = "cast<{}>(Expr({}))".format(C.type.c, beta)
        self.cpp("RDom r(0, {});".format(K))
        Y.set_shape([M, N])
        Y.set_type(C.type)
        dim_vars = self.generate_dim_vars(len(Y.shape))
        self.generate_func("norm_A")
        self.generate_func("norm_B")
        self.generate_func("norm_C")
        if transA:
            self.cpp("norm_A({}) = {}({});".format(
                JOIN_VARS(dim_vars[:2]),
                A.name,
                JOIN_VARS(dim_vars[:2][::-1])))
        else:
            self.cpp("norm_A = {};".format(A.name))
        if transB:
            self.cpp("norm_B({}) = {}({});".format(
                JOIN_VARS(dim_vars[:2]),
                B.name, JOIN_VARS(dim_vars[:2][::-1])))
        else:
            self.cpp("norm_B = {};".format(B.name))
        self.cpp("norm_C({}) = {}({});".format(
            JOIN_VARS(dim_vars),
            C.name,
            JOIN_VARS([dv if cs > 1 else "0" \
                       for dv, cs \
                       in zip(dim_vars[::-1],
                              C.shape[::-1])][::-1])))

        self.cpp("{}({}) = {}*norm_C({})+{}*sum(norm_A({})*norm_B({}));".format(
            Y.name, JOIN_VARS(dim_vars),
            beta,
            JOIN_VARS(dim_vars),
            alpha,
            JOIN_VARS([dim_vars[0], "r"]),
            JOIN_VARS(["r", dim_vars[1]])))
        
    def generate_lrn(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        alpha = 0.0001
        beta  = 0.75
        bias  = 1.0
        size  = None
        for attr in node.attribute:
            if attr.name == "alpha":
                alpha = attr.f
            if attr.name == "beta":
                beta = attr.f
            if attr.name == "bias":
                bias = attr.f
            if attr.name == "size":
                size = attr.i
 
        op.set_shape(ip.shape)
        op.set_type(ip.type)
        dim_vars = self.generate_dim_vars(len(op.shape))
        self.cpp("RDom r({}, {});".format(
            -floor((size-1)/2),
            ceil((size-1)/2)))
        self.cpp("Func padded = BoundaryConditions::constant_exterior({},0,{{{}}});".format(
            ip.name,
            JOIN_VARS(["{Expr(),Expr()}" if i != 1 else \
                       "{{0, {}}}".format(ip.shape[1]) for i in \
                       range(len(ip.shape))])))
        self.generate_func("sq_sum")
        self.cpp("sq_sum({}) = sum(pow(padded({}), 2));".format(
            JOIN_VARS(dim_vars),
            JOIN_VARS([dim_vars[0]] + ["r"] + dim_vars[2:])
        ))
        alpha = CAST(alpha, op.type.c)
        beta  = CAST(beta, op.type.c)
        bias  = CAST(bias, op.type.c)
        size  = CAST(size, op.type.c)
        self.cpp("{}({}) = {}({})/pow({}+({}/{})*sq_sum({}), {});".format(
            op.name, JOIN_VARS(dim_vars),
            ip.name, JOIN_VARS(dim_vars),
            bias, alpha, size,
            JOIN_VARS(dim_vars),
            beta))
    def generate_featuremax(self, node):
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
        if node.op_type == "Hardmax":
            self.cpp("Tuple am = argmax({}({}));".format(
                ip.name, JOIN_VARS(ip_vars)))
            self.cpp("{}({}) = cast<{}>({});".format(
                op.name, JOIN_VARS(dim_vars),
                op.type.c,
                "&&".join(["(am[{}]=={})".format(i, dv) for \
                           i, dv in enumerate(dim_vars[axis:])])))
        elif node.op_type == "LogSoftmax":
            self.generate_func("norm_ip")
            self.cpp("norm_ip({}) = {}({}) - maximum({}({}));".format(
                JOIN_VARS(dim_vars),
                ip.name, JOIN_VARS(dim_vars),
                ip.name, JOIN_VARS(ip_vars)))
            self.cpp("{}({}) = norm_ip({})-log(sum(exp(norm_ip({}))));".format(
                op.name, JOIN_VARS(dim_vars),
                JOIN_VARS(dim_vars),
                JOIN_VARS(ip_vars)))
        elif node.op_type == "Softmax":
            self.generate_func("max_x")
            self.cpp("max_x({}) = maximum({}({}));".format(
                JOIN_VARS(dim_vars),
                ip.name, JOIN_VARS(ip_vars)))
            self.generate_func("exp_x")
            self.cpp("exp_x({}) = exp({}({}) - max_x({}));".format(
                JOIN_VARS(dim_vars),
                ip.name, JOIN_VARS(dim_vars),
                JOIN_VARS(dim_vars)))
            self.cpp("{}({}) = exp_x({}) / sum(exp_x({}));".format(
                op.name, JOIN_VARS(dim_vars),
                JOIN_VARS(dim_vars),
                JOIN_VARS(ip_vars)))
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
                    HalogenType.from_onnx(attr.type).c,
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
            self.cpp("Func padded = BoundaryConditions::mirror_interior({}, {{{}}});".format(
                ip.name,
                ','.join(["{{0, {}}}".format(ips) if ips > 1 else "{Expr(),Expr()}" \
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
        if len(node.input) > 2:
            bias = self.funcs[node.input[2]]
        else:
            bias = None
        op    = self.funcs[node.output[0]]
        kernel_shape = None
        pads         = None
        padded       = False
        strides      = None
        dilations    = None
        group        = 1
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
            if attr.name == "dilations":
                dilations = list(attr.ints)
            if attr.name == "group":
                group = attr.i
        if not kernel_shape:
            kernel_shape = w.shape[2:]
        if not dilations:
            dilations = [1] * len(kernel_shape)
        if not pads:
            pads = [(0, 0) for k in kernel_shape]
        if not strides:
            strides = [1 for i in w.shape[2:]]


        op_shape = [ip.shape[0], w.shape[0]] \
                   + [floor((ips+pad[0]+pad[1]-(ks-1)*dilation-1)/stride+1) \
                      for (ips, pad, ks, stride, dilation) \
                      in zip(ip.shape[2:],
                             pads,
                             w.shape[2:],
                             strides,
                             dilations)]
        dim_vars = self.generate_dim_vars(len(ip.shape))
        self.cpp("RDom r({{{}}}, \"r\");".format(
            ','.join(["{{0,{}}}".format(i) \
                      for i in [w.shape[1]] + kernel_shape])))
        red_vars= ["r[{}]".format(i) for i in range(len(kernel_shape) + 1)]

        ip_vars = [dim_vars[0]] + ["{}+cast<int>(floor({}/{}))*{}".format(red_vars[0], dim_vars[1], op_shape[1]//group, ip.shape[1]//group)] + \
                  ["{}*{}+{}*{}-{}".format(dv, stride, rv, dilation, pad[0]) for \
                   dv, rv, pad, stride, dilation in \
                   zip(dim_vars[2:],
                       red_vars[1:],
                       pads,
                       strides,
                       dilations)]
        w_vars = [dim_vars[1]] + [red_vars[0]] + red_vars[1:]
        self.cpp()
        if padded:
            self.cpp("Func padded = BoundaryConditions::constant_exterior({}, 0, {{{}, {{Expr(), Expr()}}, {{Expr(), Expr()}}, }});".format(
                ip.name,
                ','.join(["{{0,{}}}".format(s) \
                          for i, s in enumerate(ip.shape[2:][::-1])])))
        else:
            self.cpp("Func padded = {};".format(ip.name))

        if bias:
            bias_expr = "+{}({})".format(bias.name, w_vars[0])
        else:
            bias_expr = ""
        self.cpp("{}({}) = sum(padded({}) * {}({})) {};".format(
            op.name, ','.join(dim_vars[::-1]),
            ','.join(ip_vars[::-1]),
            w.name, ','.join(w_vars[::-1]),
            bias_expr))

        op.set_shape(op_shape)
        op.set_type(ip.type)

    def generate_convT(self, node):
        ip    = self.funcs[node.input[0]]
        w     = self.funcs[node.input[1]]
        if len(node.input) > 2:
            bias = self.funcs[node.input[2]]
        else:
            bias = None
        op    = self.funcs[node.output[0]]
        kernel_shape = None
        dilations    = None
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
        if not dilations:
            dilations = [1 for i in w.shape[2:]]
        if not kernel_shape:
            kernel_shape = w.shape[2:]
        if not pads:
            pads = [(0, 0) for w in kernel_shape]
        if not op_pads:
            op_pads = [0 for j in kernel_shape]
        if op_shape: # Output shape explicit
            if len(op_shape) < len(ip.shape):
                op_shape = [ip.shape[0], w.shape[1]] + op_shape
            total_padding = [stride*(ops-1)+op_pad+ks-ips \
                             for (ips, ops, op_pad, ks, stride) in \
                             zip(ip.shape[2:],
                                 op_shape[2:],
                                 op_pads,
                                 kernel_shape,
                                 strides)]
            if auto_pad:
                if auto_pad != "SAME_UPPER":
                    pads = [(tp//2, tp-tp//2) for tp in total_padding]
                else:
                    pads = [(tp-tp//2, tp//2) for tp in total_padding]
        else: # Infer output shape 
            op_shape = [ip.shape[0], w.shape[1]] + \
                       [stride*(ips-1)+op_pad+ks-pad[0]-pad[1] \
                        for (ips, op_pad, ks, stride, pad) in
                        zip(ip.shape[2:],
                            op_pads,
                            kernel_shape,
                            strides,
                            pads)]

        dim_vars = self.generate_dim_vars(len(ip.shape))
        self.cpp("RDom r({{{}}}, \"r\");".format(
            ','.join(["{{0,{}}}".format(i) \
                      for i, st in zip(ip.shape[1:],
                                   [1] + strides)])))
        red_vars= ["r[{}]".format(i) for i in range(len(kernel_shape) + 1)]

        self.cpp("Func padded = BoundaryConditions::constant_exterior({}, 0, {{{}, {{Expr(), Expr()}}, {{Expr(), Expr()}}, }});".format(
            w.name,
            ','.join(["{{0,{}}}".format(s) \
                      for i, s in enumerate(w.shape[2:][::-1])])))
        self.cpp("Func padded_ip = BoundaryConditions::constant_exterior({}, 0, {{{}, {{Expr(),Expr()}},{{Expr(),Expr()}},}});".format(
            ip.name,
            JOIN_VARS(["{{0,{}}}".format(s) for s in ip.shape[2:]])))
        dilated_op_vars = dim_vars
        dilated_expr = ["(({}%{})==0)".format(dv, dil) \
                        for dil, dv in zip(dilations, dim_vars[2:])]
        dilated_ip_vars = ["cast<int>(floor({}/{}))".format(dv, dil) \
                           for dil, dv in zip(dilations, dim_vars[2:])]
        self.generate_func("dilated");
        self.cpp("dilated({}) = select({}, padded({}), 0);".format(
            JOIN_VARS(dilated_op_vars),
            "&&".join(dilated_expr[::-1]),
            JOIN_VARS(dim_vars[:2] + dilated_ip_vars)))
        self.cpp()

        ip_vars = [dim_vars[0]] + [red_vars[0]] + \
                  ["cast<int>(floor(({0}-{4}*{5}+{2})/{1}))".format(dv, st, pad[0], op_pad, rv, dil) for dv, st, pad, op_pad, rv, dil \
                   in zip(dim_vars[2:],
                          strides,
                          pads,
                          op_pads,
                          red_vars[1:],
                          dilations)]
        w_vars = [red_vars[0]] + [dim_vars[1]] + \
                 ["{}*{}".format(rv, dil) for \
                  dv, rv, pad, dil, stride in \
                  zip(dim_vars[2:],
                      red_vars[1:],
                      pads,
                      dilations,
                      strides)]
        sel_expr = ["((({0}-{2}*{3}+{5})%{1})==0)".format(dv, st, rv, dil, op_pad, pad[0]) for \
                    dv, rv, st, dil, op_pad, pad in \
                    zip(dim_vars[2:],
                        red_vars[1:],
                        strides,
                        dilations,
                        op_pads,
                        pads)]
        self.cpp()

        if bias:
            bias_expr = "+{}({})".format(bias.name, dim_vars[1])
        else:
            bias_expr = ""
        self.cpp("{0}({1}) = sum(select({2}, {3}({4}), 0) * dilated({5})) {6};".format(
            op.name, JOIN_VARS(dim_vars),
            "&&".join(sel_expr[::-1]),
            "padded_ip", JOIN_VARS(ip_vars),
            JOIN_VARS(w_vars),
            bias_expr))
        op.set_shape(op_shape)
        op.set_type(ip.type)

    def generate_slice(self, node):
        ip = self.funcs[node.input[0]]
        op = self.funcs[node.output[0]]
        starts = None
        ends = None
        axes = None
        for attr in node.attribute:
            if attr.name == "axes":
                axes = list(attr.ints)
            if attr.name == "ends":
                ends = list(attr.ints)
            if attr.name == "starts":
                starts = list(attr.ints)
        if not axes:
            axes = list(range(len(starts)))
        op_shape = ip.shape
        st_dict = {}
        for (i, s, e) in zip(axes, starts, ends):
            s = max(0, min(ip.shape[i], s))
            if e < 0:
                e = ip.shape[i] + e
            e = min(ip.shape[i], e)
            op_shape[i] = e - s
            st_dict[i] = s
        op.set_shape(op_shape)
        op.set_type(ip.type)
        dim_vars = self.generate_dim_vars(len(op.shape))
        self.cpp("{}({}) = {}({});".format(
            op.name, JOIN_VARS(dim_vars),
            ip.name, JOIN_VARS(["{}+{}".format(dv, st_dict[i]) if i in st_dict else dv \
                                for i, dv in \
                                enumerate(dim_vars)])))
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
        id    = None
        if len(node.output) > 1:
            id= self.funcs[node.output[1]]
        pool_shape        = None
        count_include_pad = False
        pads              = None
        padded            = False
        auto_pad          = None
        strides           = None
        storage_order     = 0
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
            if attr.name == "storage_order":
                storage_order = attr.i
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
        if node.op_type in ["GlobalMaxPool", "MaxPool"]:
            pad_const = ip.type.c_min
        else:
            pad_const = 0
        op_shape = [floor((ips+pad[0]+pad[1]-ks)/stride+1) \
                    if pad else ips \
                    for (ips, pad, ks, stride) \
                    in zip(ip.shape,
                           [None] * n_ign_dims + pads,
                           [None] * n_ign_dims + pool_shape,
                           [None] * n_ign_dims + strides)]

        op.set_shape(op_shape)
        op.set_type(ip.type)
        
        if padded:
            self.cpp("Func padded = BoundaryConditions::constant_exterior({}, {}, {{{}}});".format(
                ip.name,
                pad_const,
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
            if id:
                id.set_shape(op.shape)
                id.set_type(HalogenType.from_c("int64_t"))
                self.generate_func("maxed")
                self.cpp("maxed({}) = argmax(padded({}));".format(
                    JOIN_VARS(dim_vars),
                    JOIN_VARS(ip_vars)))
                prod = ip.shape[::-1]
                prod = [int(np.prod(prod[:i])) for i in range(len(prod))]
                if storage_order == 1:
                    prod[:2] = prod[:2][::-1]
                maxed_vars = ["({}*{}+maxed({})[{}]-{})*{}".format(dv,
                                                                   st,
                                                                   JOIN_VARS(dim_vars),
                                                                   i-n_ign_dims,
                                                                   pad[0],
                                                                   prod)
                              if rv else "{}*{}".format(dv, prod) \
                              for i, (dv, rv, st, pad, prod) in enumerate(zip(
                                      dim_vars,
                                      [None]*n_ign_dims + red_vars,
                                      [None]*n_ign_dims + strides,
                                      [None]*n_ign_dims + pads,
                                      prod[::-1]))]

                self.cpp("{}({}) = cast<int64_t>({});".format(id.name,
                                               JOIN_VARS(dim_vars),
                                               '+'.join(maxed_vars)))
            self.cpp("{}({}) = maximum(padded({}));".format(
                op.name, ','.join(dim_vars[::-1]),
                ','.join(ip_vars[::-1])))

        return

    def generate_node(self, nidx, node):
        for op in node.output:
            if op not in self.funcs:
                f_name = "f_" + op.replace('/', '').replace('-','')
                self.generate_func(f_name)
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
            self.generate_unary_expr(node,"select({0} < 0, cast<{2}>(Expr({1}) * (exp({0}) - Expr(1.))), {0})")
        elif node.op_type == "Exp":
            self.generate_unary_expr(node, "exp({})")
        elif node.op_type == "Floor":
            self.generate_unary_expr(node, "floor({})")
        elif node.op_type == "HardSigmoid":
            self.generate_unary_expr(node, "clamp({}*{}+{},0,1)")
        elif node.op_type == "Identity":
            self.generate_unary_expr(node, "{}")
        elif node.op_type == "LeakyRelu":
            self.generate_unary_expr(node, "select({0}<0,{1}*{0},{0})")
        elif node.op_type == "Log":
            self.generate_unary_expr(node, "log({})")
        elif node.op_type == "Neg":
            self.generate_unary_expr(node, "-{}")
        elif node.op_type == "Not":
            self.generate_unary_expr(node, "cast<int8_t>({} == 0)")
        elif node.op_type == "Reciprocal":
            self.generate_unary_expr(node, "1/{}")
        elif node.op_type == "Relu":
            self.generate_unary_expr(node, "select({0}>0, {0}, 0)")
        elif node.op_type == "Selu":
            self.generate_unary_expr(node, "select({0}<0, {2}*({1}*exp({0}) - {1}), {2}*{0})")
        elif node.op_type == "Sigmoid":
            self.generate_unary_expr(node, "1 / (1 + exp(-{}))")
        elif node.op_type == "Sin":
            self.generate_unary_expr(node, "sin({})")
        elif node.op_type == "Sinh":
            self.generate_unary_expr(node, "sinh({})")
        elif node.op_type == "Softplus":
            self.generate_unary_expr(node, "log(exp({}) + 1)")
        elif node.op_type == "Softsign":
            self.generate_unary_expr(node, "{0} / (1 + abs({0}))")
        elif node.op_type == "Sqrt":
            self.generate_unary_expr(node, "sqrt({})")
        elif node.op_type == "Tan":
            self.generate_unary_expr(node, "tan({})")
        elif node.op_type == "Tanh":
            self.generate_unary_expr(node, "tanh({})")
        elif node.op_type == "ThresholdedRelu":
            self.generate_unary_expr(node, "select({0} > {1}, {0}, 0)")
        elif node.op_type == "Add":
            self.generate_bin_expr(node, "{}+{}")
        elif node.op_type == "And":
            self.generate_bin_expr(node, "{}&{}")
        elif node.op_type == "Div":
            self.generate_bin_expr(node, "{}/{}")
        elif node.op_type == "Equal":
            self.generate_bin_expr(node, "{}=={}", HalogenType.from_c("int8_t"))
        elif node.op_type == "Greater":
            self.generate_bin_expr(node, "{}>{}", HalogenType.from_c("int8_t"))
        elif node.op_type == "Less":
            self.generate_bin_expr(node, "{}<{}", HalogenType.from_c("int8_t"))
        elif node.op_type == "Mul":
            self.generate_bin_expr(node, "{}*{}")
        elif node.op_type == "Or":
            self.generate_bin_expr(node, "{}|{}", HalogenType.from_c("int8_t"))
        elif node.op_type == "Pow":
            self.generate_bin_expr(node, "pow({0},{1})")
        elif node.op_type == "Sub":
            self.generate_bin_expr(node, "{}-{}")
        elif node.op_type == "Xor":
            self.generate_bin_expr(node, "{}^{}")
        elif node.op_type == "Max":
            self.generate_var_expr(node)
        elif node.op_type == "Mean":
            self.generate_var_expr(node)
        elif node.op_type == "Min":
            self.generate_var_expr(node)
        elif node.op_type == "Sum":
            self.generate_var_expr(node)
        elif node.op_type == "ArgMax":
            self.generate_red_expr(node, "argmax", "int64_t")
        elif node.op_type == "ArgMin":
            self.generate_red_expr(node, "argmin", "int64_t")
        elif node.op_type == "ReduceL1":
            self.generate_redl_expr(node, "sum(abs({}))")
        elif node.op_type == "ReduceL2":
            self.generate_redl_expr(node, "sqrt(sum(pow({},2)))")
        elif node.op_type == "ReduceLogSum":
            self.generate_redl_expr(node, "log(sum({}))")
        elif node.op_type == "ReduceLogSumExp":
            self.generate_redl_expr(node, "log(sum(exp({})))")
        elif node.op_type == "ReduceMax":
            self.generate_redl_expr(node, "maximum({})")
        elif node.op_type == "ReduceMean":
            self.generate_redl_expr(node, "sum({})/{}")
        elif node.op_type == "ReduceMin":
            self.generate_redl_expr(node, "minimum({})")
        elif node.op_type == "ReduceProd":
            self.generate_redl_expr(node, "product({})")
        elif node.op_type == "ReduceSum":
            self.generate_redl_expr(node, "sum({})")
        elif node.op_type == "ReduceSumSquare":
            self.generate_redl_expr(node, "sum(pow({},2))")
        elif node.op_type == "BatchNormalization":
            self.generate_bn(node)
        elif node.op_type == "InstanceNormalization":
            self.generate_in(node)
        elif node.op_type == "AveragePool":
            self.generate_pool(node)
        elif node.op_type == "GlobalAveragePool":
            self.generate_pool(node)
        elif node.op_type == "GlobalMaxPool":
            self.generate_pool(node)
        elif node.op_type == "MaxPool":
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
        elif node.op_type == "PRelu":
            self.generate_prelu_expr(node)
        elif node.op_type == "Flatten":
            self.generate_flatten(node)
        elif node.op_type == "Gather":
            self.generate_gather(node)
        elif node.op_type == "Gemm":
            self.generate_gemm(node)
        elif node.op_type == "MatMul":
            self.generate_matmul(node)
        elif node.op_type == "Hardmax":
            self.generate_featuremax(node)
        elif node.op_type == "LogSoftmax":
            self.generate_featuremax(node)
        elif node.op_type == "Softmax":
            self.generate_featuremax(node)
        elif node.op_type == "LRN":
            self.generate_lrn(node)
        elif node.op_type == "Constant":
            self.generate_constant(node)
        elif node.op_type == "Shape":
            self.generate_shape(node)
        elif node.op_type == "Size":
            self.generate_size(node)
        elif node.op_type == "Slice":
            self.generate_slice(node)
        elif node.op_type == "Split":
            self.generate_split(node)
        elif node.op_type == "Squeeze":
            self.generate_squeeze(node)
        elif node.op_type == "Transpose":
            self.generate_transpose(node)
        elif node.op_type == "Unsqueeze":
            self.generate_unsqueeze(node)
        elif node.op_type == "Upsample":
            raise NotImplementedError
        elif node.op_type == "TopK":
            raise NotImplementedError
        elif node.op_type == "Tile":
            raise NotImplementedError
        elif node.op_type == "Reshape":
            self.generate_reshape(node)
        elif node.op_type == "GRU":
            raise NotImplementedError
        elif node.op_type == "RNN":
            raise NotImplementedError
        elif node.op_type == "Expand":
            raise NotImplementedError
        else:
            print()
            print("unhandled node ", node.op_type)
            raise NotImplementedError
        self.cpp("}", -1)

        for op in node.output:
            try:
                self.cpp("// {} is {}{}".format(op, self.funcs[op].type.c,
                                              self.funcs[op].shape))
            except:
                print(node)
                with open("generated/halogen_generator.cpp", "w") as f:
                    f.write(self.halogen_str)
                assert(False)


