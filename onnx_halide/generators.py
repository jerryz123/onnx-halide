from math import floor, ceil
import numpy as np
from .tensortypes import HalideObj, HalogenType

class CppGenerator:
    def __init__(self):
        self.code = []
        self.indent = 0

    def __call__(self, s="", incr=0):
        nstr = ""
        if incr < 0:
            self.indent += incr
        if s:
            nstr += "  " * self.indent
        nstr += "{}\n".format(s)
        if incr > 0:
            self.indent += incr
        self.code.append((nstr, self.indent))
    def block(self, startstr="", endstr="", incr=0):
        gen = CppGenerator()
        self.__call__(startstr, incr)
        self.code.append((gen, self.indent))
        self.__call__(endstr, -1*incr)
        return gen
    def get_code(self):
        code = []
        for l, indent in self.code:
            if type(l) == str:
                code.append(l)
            else:
                for li in l.get_code():
                    code.append("  " * indent + li)
        return code

    def write(self, fname):
        with open(fname, 'w') as f:
            for l in self.get_code():
                f.write(l)

class NodeGenerator:
    op_type     = ""
    attr_fields = {}
    @classmethod
    def match_class(cls, node):
        if node.op_type == cls.op_type:
            return cls
        for subclass in cls.__subclasses__():
            m = subclass.match_class(node)
            if m:
                return m
        return False

    def __init__(self, node, alg_generator, funcs):
        # Some strangeness here, find the generator which matches our
        # node's operator type, and set our type to that class
        self.__class__ = NodeGenerator.match_class(node)
        assert(self.__class__ and self.op_type)
        self.alg = alg_generator

        # Find input funcs
        n_ip = len(node.input)
        for i in range(n_ip):
            setattr(self, "ip{}".format(i),
                    funcs[node.input[i]])

        # Find output funcs
        n_op = len(node.output)
        for i in range(n_op):
            setattr(self, "op{}".format(i),
                    funcs[node.output[i]])

        # Find attrs
        for attr_name, (attr_k, attr_v, attr_def) in self.attr_fields.items():
            for attr in node.attribute:
                if attr.name == attr_k:
                    v = getattr(attr, attr_v)
                    if attr_v == "ints":
                        v = list(v)
                    setattr(self, "{}_".format(attr_name),
                            v)
                    break
            else:
                setattr(self, "{}_".format(attr_name),
                        attr_def)

        self.pp_attrs()
        self.infer_types()
        self.infer_shapes()
        self.infer_dim_vars()

    def pp_attrs(self):
        pass

    def infer_types(self):
        self.op0.set_type(self.ip0.type)

    def infer_shapes(self):
        self.op0.set_shape(self.ip0.shape)

    def infer_dim_vars(self):
        self.n_dim_vars = self.op0.dims

    def generate_alg(self, dim_vars):
        raise NotImplementedError

    def generate_sched(self):
        pass

    def generate_funcref(self, func, dim_vars):
        assert(type(func) == str or len(dim_vars) == func.dims)
        return "{}({})".format(func.name if type(func) == HalideObj else func,
                               ','.join(dim_vars[::-1]))

    def generate_asn(self, op, ip):
        self.alg("{} = {};".format(op, ip))

    def generate_funcdecl(self, name, rhs=""):
        name = "{}_{}".format(self.op0.name, name)
        if rhs:
            rhs = " = {}".format(rhs)
        self.alg("Func {}{};".format(name, rhs))
        return name

    def generate_padded(self, name, ip, pad_const, pad_doms):
        return self.generate_funcdecl(
            name,
            "constant_exterior({}, {}, {{{}}})".format(
                ip if type(ip) == str else ip.name,
                pad_const,
                ','.join(["{{{},{}}}".format(a, b) for a, b in pad_doms][::-1])))
    def generate_cast(self, type, expr):
        return "cast<{}>({})".format(type.c, expr)

    def generate_rdom(self, name, ranges):
        rdom_name = "{}_{}".format(self.op0.name, name)
        self.alg("RDom {}({});".format(rdom_name,
                                       ','.join(["{},{}".format(a,b) \
                                                 for a, b in ranges])))
        return ["{}[{}]".format(rdom_name, i) for i in range(len(ranges))]

class UnaryGenerator(NodeGenerator):
    def generate_alg(self, dim_vars):
        ip0_expr  = self.generate_funcref(self.ip0,
                                          dim_vars[:self.ip0.dims])
        op0_expr  = self.generate_funcref(self.op0,
                                          dim_vars[:self.ip0.dims])
        unop_expr = self.expr.format(ip0_expr)
        self.generate_asn(op0_expr, unop_expr)


class AbsGenerator(UnaryGenerator):
    op_type = "Abs"
    expr    = "abs({})"

class AcosGenerator(UnaryGenerator):
    op_type = "Acos"
    expr    = "acos({})"

class AsinGenerator(UnaryGenerator):
    op_type = "Asin"
    expr    = "asin({})"

class AtanGenerator(UnaryGenerator):
    op_type = "Atan"
    expr    = "atan({})"

class BinaryGenerator(NodeGenerator):
    def infer_shapes(self):
        dims = max(self.ip0.dims, self.ip1.dims)
        self.op0.set_shape(
            [self.ip1.shape[-i] if i > self.ip0.dims else
             (self.ip0.shape[-i] if i > self.ip1.dims else
              max(self.ip0.shape[-i], self.ip1.shape[-i])) \
             for i in range(1, dims+1)][::-1])

    def generate_alg(self, dim_vars):
        ip0_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(self.ip0.shape,
                            dim_vars[-len(self.ip0.shape):])]
        ip0_expr     = self.generate_funcref(self.ip0, ip0_dim_vars)
        ip1_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(self.ip1.shape,
                            dim_vars[-len(self.ip1.shape):])]
        ip1_expr     = self.generate_funcref(self.ip1, ip1_dim_vars)

        expr         = self.expr.format(ip0_expr, ip1_expr)

        op0_expr     = self.generate_funcref(self.op0, dim_vars)
        self.generate_asn(op0_expr, expr)

class AddGenerator(BinaryGenerator):
    op_type = "Add"
    expr    = "{}+{}"

class AndGenerator(BinaryGenerator):
    op_type = "And"
    expr    = "{}&{}"

class ArgMGenerator(NodeGenerator):
    attr_fields = {"keepdims":("keepdims", "i", 1),
                   "axis"    :("axis"    , "i", 0)}

    def infer_shapes(self):
        op_shape = self.ip0.shape
        if self.keepdims_:
            op_shape[self.axis_] = 1
        else:
            op_shape.pop(self.axis_)
        self.op0.set_shape(op_shape)

    def infer_types(self):
        self.op0.set_type(HalogenType.from_c("int64_t"))

    def infer_dim_vars(self):
        self.n_dim_vars = self.ip0.dims

    def generate_alg(self, dim_vars):

        red_vars = self.generate_rdom("r",
                                      [[0, self.ip0.shape[self.axis_]]])

        op_dim_vars = list(dim_vars) if self.keepdims_ else \
                         [dvar for i, dvar in enumerate(dim_vars) \
                          if i != self.axis_]

        ip_dim_vars = [(dvar if i != self.axis_ \
                        else red_vars[0]) \
                        for i, dvar in enumerate(dim_vars)]

        op_expr = self.generate_funcref(self.op0, op_dim_vars)
        ip_expr = self.generate_funcref(self.ip0, ip_dim_vars)

        expr = "{}({})[0]".format(self.argm_type, ip_expr)
        expr = self.generate_cast(self.op0.type, expr)
        self.generate_asn(op_expr, expr)

class ArgMaxGenerator(ArgMGenerator):
    op_type   = "ArgMax"
    argm_type = "argmax"

class ArgMinGenerator(ArgMGenerator):
    op_type   = "ArgMin"
    argm_type = "argmin"

class PoolGenerator(NodeGenerator):
    attr_fields = {"kernel_shape"     :("kernel_shape"     , "ints", None),
                   "count_include_pad":("count_include_pad", "i"   , 0),
                   "pads"             :("pads"             , "ints", None),
                   "auto_pad"         :("auto_pad"         , "s"   , ""),
                   "strides"          :("strides"          , "ints", None),
                   "storage_order"    :("storage_order"    , "i"   , 0)}

    def pp_attrs(self):
        self.pool_shape_   = list(self.kernel_shape_) or self.ip0.shape[2:]
        self.strides_      = self.strides_ or [1 for ks in self.pool_shape_]
        if type(self.auto_pad_) != str:
            self.auto_pad_ = self.auto_pad_.decode()
        if self.pads_:
            li             = len(self.pads_)//2
            self.pads_     = list(zip(self.pads_[:li],
                                      self.pads_[li:]))
        else:
            if self.auto_pad_ == "SAME_UPPER":
                self.pads_ = [(floor((ks-1)/2), ceil((ks-1)/2)) \
                              for ks in self.pool_shape_]
            elif self.auto_pad_ == "SAME_LOWER":
                self.pads_ = [(ceil((ks-1)/2), floor((ks-1)/2)) \
                              for ks in self.pool_shape_]
            else:
                self.pads_ = [(0, 0) for ks in self.pool_shape_]
        self.padded_       = sum(map(sum, self.pads_)) > 0
        self.n_ign_dims_   = len(self.ip0.shape) - len(self.pool_shape_)

    def infer_shapes(self):
        self.op0.set_shape([floor((ips+pad[0]+pad[1]-ks)/stride+1) \
                            if pad else ips \
                            for (ips, pad, ks, stride) \
                            in zip(self.ip0.shape,
                                   [()]*self.n_ign_dims_ + self.pads_,
                                   [()]*self.n_ign_dims_ + self.pool_shape_,
                                   [()]*self.n_ign_dims_ + self.strides_)])

    def infer_dim_vars(self):
        self.n_dim_vars = len(self.ip0.shape)

    def generate_alg(self, dim_vars):

        red_vars = self.generate_rdom("r",
                           [(0, i) for i in self.pool_shape_])

        ip_vars = ["{}*{}+{}-{}".format(dv, st, rv, pad[0]) \
                   if rv else dv \
                   for (dv, rv, st, pad) in zip(
                           dim_vars,
                           [()]*self.n_ign_dims_ + red_vars,
                           [()]*self.n_ign_dims_ + self.strides_,
                           [()]*self.n_ign_dims_ + self.pads_)]

        if self.padded_:
            padded = self.generate_padded(
                "pad", self.ip0,
                self.pad_const,
                [(0, s) \
                 if i >= self.n_ign_dims_ else \
                 ("Expr()", "Expr()") \
                 for i, s in enumerate(self.ip0.shape)])
        else:
            padded = self.generate_funcdecl("pad", self.ip0.name)


        lhs = self.generate_funcref(self.op0, dim_vars)
        rhs = self.generate_pool_rhs(dim_vars, red_vars, ip_vars).format(
            self.generate_funcref(padded,
                                  ip_vars))
        self.generate_asn(lhs, rhs)

class AveragePoolGenerator(PoolGenerator):
    op_type   = "AveragePool"
    pad_const = "0"
    def generate_pool_rhs(self, dim_vars, red_vars, ip_vars):
        count_func = self.generate_funcdecl("count")

        count_vars  = dim_vars[-len(red_vars):]
        count_expr  = self.generate_funcref(count_func,
                                            count_vars)
        if self.count_include_pad_:
            count_rhs = np.prod(self.pool_shape_)
        else:
            ones_func = self.generate_funcdecl("ones")
            self.generate_asn(
                self.generate_funcref(ones_func,
                                      count_vars),
                "1")
            padded_ones = self.generate_padded(
                "pad_ones", ones_func, "0", 
                [(0, s) for s in \
                 self.ip0.shape[-len(red_vars):]])

            pad_ones_expr = self.generate_funcref(padded_ones,
                                                  ip_vars[-len(red_vars):])
            count_rhs = "sum({})".format(pad_ones_expr)
        self.generate_asn(count_expr, count_rhs)
        rhs_expr = "sum({{}}) / {}".format(self.generate_funcref(
            count_func,
            dim_vars[-len(red_vars):]))
        return rhs_expr

class ConvGenerator(NodeGenerator):
    op_type     = "Conv"
    attr_fields = {"kernel_shape": ("kernel_shape", "ints", None),
                   "pads"        : ("pads"        , "ints", None),
                   "strides"     : ("strides"     , "ints", None),
                   "dilations"   : ("dilations"   , "ints", None),
                   "group"       : ("group"       , "i"   , 1)}
    def pp_attrs(self):
        self.w = self.ip1

        if self.pads_:
            li = len(self.pads_) // 2
            self.pads_      = list(zip(self.pads_[:li], self.pads_[li:]))
        self.kernel_shape_  = self.kernel_shape_ or self.w.shape[2:]
        self.dilations_     = self.dilations_ or [1] * len(self.kernel_shape_)
        self.pads_          = self.pads_ or [(0, 0)] * len(self.kernel_shape_)
        self.strides_       = self.strides_ or [1] * len(self.kernel_shape_)
        self.padded_        = sum(map(sum, self.pads_)) > 0
        self.bias_          = hasattr(self, "ip2")

    def infer_shapes(self):
        self.op0.set_shape([self.ip0.shape[0], self.w.shape[0]] + \
                           [floor((ips+pad[0]+pad[1]-(ks-1)*dilation-1)/stride+1) \
                            for (ips, pad, ks, stride, dilation) \
                            in zip(self.ip0.shape[2:],
                                   self.pads_,
                                   self.w.shape[2:],
                                   self.strides_,
                                   self.dilations_)])

    def generate_alg(self, dim_vars):
        red_vars = self.generate_rdom(
            "r",
            [(0, i) for i in [self.w.shape[1]] + self.kernel_shape_])
        ip_vars  = [dim_vars[0], "{}+cast<int>(floor({}/{}))*{}".format(
            red_vars[0],
            dim_vars[1],
            self.op0.shape[1]//self.group_, self.ip0.shape[1]//self.group_)] + \
            ["{}*{}+{}*{}-{}".format(dv, stride, rv, dilation, pad[0]) for \
             dv, rv, pad, stride, dilation in \
             zip(dim_vars[2:],
                 red_vars[1:],
                 self.pads_,
                 self.strides_,
                 self.dilations_)]
        w_vars  = [dim_vars[1]] + red_vars

        if self.padded_:
            padded = self.generate_padded("padded",
                                          self.ip0.name,
                                          0,
                                          [("Expr()", "Expr()")] * 2 + \
                                          [(0, s) for s in self.ip0.shape[2:]])
        else:
            padded = self.generate_funcdecl("padded", rhs=self.ip0.name)
        padded_expr = self.generate_funcref(padded, ip_vars)
        
        if self.bias_:
            bias_expr = self.generate_funcref(self.ip2, dim_vars[1])
        else:
            bias_expr = "0"

        w_expr = self.generate_funcref(self.w, w_vars)

        lhs = self.generate_funcref(self.op0, dim_vars)
        rhs = "sum({}*{})+{}".format(padded_expr, w_expr, bias_expr)
        self.generate_asn(lhs, rhs)
