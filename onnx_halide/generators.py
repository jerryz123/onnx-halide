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

    def __init__(self, node, alg_generator, sched_generator, funcs, init_data):
        # Some strangeness here, find the generator which matches our
        # node's operator type, and set our type to that class
        self.__class__ = NodeGenerator.match_class(node)
        self.sch = sched_generator
        assert(self.__class__ and self.op_type)
        self.alg = alg_generator
        self.alg("// {}".format(self.op_type))
        # Find input funcs
        n_ip = len(node.input)
        self.ips = []
        for i in range(n_ip):
            setattr(self, "ip{}".format(i),
                    funcs[node.input[i]])
            self.ips.append(funcs[node.input[i]])
            if str(node.input[i]) in init_data:
                self.ips[i].data = init_data[node.input[i]]


        # Find output funcs
        n_op = len(node.output)
        self.ops = []
        for i in range(n_op):
            if node.output[i] not in funcs:
                f_name = "f_" + node.output[i].replace('/','').replace('-','')
                funcs[node.output[i]] = HalideObj(f_name,)
                self.alg("Func {};".format(f_name))
            self.ops.append(funcs[node.output[i]])
            setattr(self, "op{}".format(i),
                    funcs[node.output[i]])

        # Find attrs
        for attr_name, (attr_k, attr_v, attr_def) in self.attr_fields.items():
            for attr in node.attribute:
                if attr.name == attr_k:
                    v = getattr(attr, attr_v)
                    if attr_v == "ints":
                        v = list(v)
                    elif attr_v == "s":
                        v = v.decode()
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
        for op in self.ops:
            self.sch("{}.compute_root();".format(op.name))

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
    def generate_cast(self, type_, expr):
        if type(expr) != str:
            return "cast<{}>(Expr({}))".format(type_.c, expr)
        else:
            return "cast<{}>({})".format(type_.c, expr)

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

class CosGenerator(UnaryGenerator):
    op_type = "Cos"
    expr    = "cos({})"

class CastGenerator(UnaryGenerator):
    op_type = "Cast"
    attr_fields = {"to":("to", "i", None)}
    def pp_attrs(self):
        if self.to_:
            self.to_ = HalogenType.from_onnx(self.to_)
        else:
            self.to_ = self.ip0.type
    def infer_types(self):
        self.op0.set_type(self.to_)
    @property
    def expr(self):
        return self.generate_cast(self.to_, "{}")

class CeilGenerator(UnaryGenerator):
    op_type = "Ceil"
    expr    = "ceil({})"

class ClipGenerator(UnaryGenerator):
    op_type = "Clip"
    attr_fields = {"min_v": ("min", "f", None),
                   "max_v": ("max", "f", None)}
    def pp_attrs(self):
        self.min_v_ = "Expr({})".format(self.min_v_) if self.min_v_ is not None \
                      else self.ip0.type.c_min
        self.max_v_ = "Expr({})".format(self.max_v_) if self.max_v_ is not None \
                      else self.ip0.type.c_max
    @property
    def expr(self):
        return "clamp({{}}, {}, {})".format(self.min_v_,
                                            self.max_v_)

class DropoutGenerator(UnaryGenerator):
    op_type = "Dropout"
    expr    = "{}"

class EluGenerator(UnaryGenerator):
    op_type = "Elu"
    attr_fields = {"alpha":("alpha", "f", 1.0)}
    @property
    def expr(self):
        return "select({{0}}<0,cast<{}>(Expr({})*(exp({{0}})-Expr(1.))),{{0}})".format(
            self.ip0.type.c, self.alpha_)

class ExpGenerator(UnaryGenerator):
    op_type = "Exp"
    expr    = "exp({})"

class FloorGenerator(UnaryGenerator):
    op_type = "Floor"
    expr    = "floor({})"

class HardSigmoidGenerator(UnaryGenerator):
    op_type = "HardSigmoid"
    attr_fields = {"alpha":("alpha", "f", 0.2),
                   "beta":("beta", "f", 0.5)}
    @property
    def expr(self):
        return "clamp({{}}*{0}+{1},0,1)".format(
            self.generate_cast(self.op0.type, "Expr({})".format(self.alpha_)),
            self.generate_cast(self.op0.type, "Expr({})".format(self.beta_)))

class IdentityGenerator(UnaryGenerator):
    op_type = "Identity"
    expr    = "{}"

class LeakyReluGenerator(UnaryGenerator):
    op_type = "LeakyRelu"
    attr_fields = {"alpha":("alpha", "f", 0.01)}
    @property
    def expr(self):
        return "select({{0}}<0,{}*{{0}},{{0}})".format(
            self.generate_cast(self.op0.type, "Expr({})".format(self.alpha_)))

class LogGenerator(UnaryGenerator):
    op_type = "Log"
    expr    = "log({})"

class NegGenerator(UnaryGenerator):
    op_type = "Neg"
    expr    = "-{}"

class NotGenerator(UnaryGenerator):
    op_type = "Not"
    expr    = "cast<int8_t>({}==0)"

class ReciprocalGenerator(UnaryGenerator):
    op_type = "Reciprocal"
    expr    = "1/{}"

class ReluGenerator(UnaryGenerator):
    op_type = "Relu"
    expr    = "select({0}>0,{0},0)"

class SeluGenerator(UnaryGenerator):
    op_type = "Selu"
    attr_fields = {"alpha":("alpha","f",1.67326),
                   "gamma":("gamma","f",1.0507)}
    @property
    def expr(self):
        return "select({{0}}<0,{0}*({1}*exp({{0}})-{1}),{0}*{{0}})".format(
            self.generate_cast(self.op0.type, self.gamma_),
            self.generate_cast(self.op0.type, self.alpha_)
            )

class SigmoidGenerator(UnaryGenerator):
    op_type = "Sigmoid"
    expr    = "(1/(1+exp(-{})))"

class SinGenerator(UnaryGenerator):
    op_type = "Sin"
    expr    = "sin({})"

class SoftplusGenerator(UnaryGenerator):
    op_type = "Softplus"
    expr    = "log(exp({})+1)"

class SoftsignGenerator(UnaryGenerator):
    op_type = "Softsign"
    expr    = "{0}/(1+abs({0}))"

class SqrtGenerator(UnaryGenerator):
    op_type = "Sqrt"
    expr    = "sqrt({})"

class TanGenerator(UnaryGenerator):
    op_type = "Tan"
    expr    = "tan({})"

class TanhGenerator(UnaryGenerator):
    op_type = "Tanh"
    expr    = "tanh({})"

class ThrehReluGenerator(UnaryGenerator):
    op_type = "ThresholdedRelu"
    attr_fields = {"alpha":("alpha", "f", 1.0)}
    @property
    def expr(self):
        return "select({{0}}>{},{{0}},0)".format(
            self.generate_cast(self.op0.type, self.alpha_))
            

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

class DivGenerator(BinaryGenerator):
    op_type = "Div"
    expr    = "{}/{}"

class MulGenerator(BinaryGenerator):
    op_type = "Mul"
    expr    = "{}*{}"

class PowGenerator(BinaryGenerator):
    op_type = "Pow"
    expr    = "pow({},{})"

class SubGenerator(BinaryGenerator):
    op_type = "Sub"
    expr    = "{}-{}"

class BinOpGenerator(BinaryGenerator):
    def infer_types(self):
        self.op0.set_type(HalogenType.from_c("int8_t"))

class EqualGenerator(BinOpGenerator):
    op_type = "Equal"
    expr    = "cast<int8_t>({}=={})"

class GreaterGenerator(BinOpGenerator):
    op_type = "Greater"
    expr    = "cast<int8_t>({}>{})"

class LessGenerator(BinOpGenerator):
    op_type = "Less"
    expr    = "cast<int8_t>({}<{})"

class OrGenerator(BinOpGenerator):
    op_type = "Or"
    expr    = "cast<int8_t>({}|{})"

class XorGenerator(BinOpGenerator):
    op_type = "Xor"
    expr    = "cast<int8_t>({}^{})"

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
        self.pool_shape_   = list(self.kernel_shape_) if self.kernel_shape_ else self.ip0.shape[2:]
        self.strides_      = self.strides_ or [1 for ks in self.pool_shape_]
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
        rhs = self.generate_pool_rhs(dim_vars, red_vars, ip_vars,
                                     self.generate_funcref(padded,
                                                           ip_vars))
        self.generate_asn(lhs, rhs)

class AveragePoolGenerator(PoolGenerator):
    op_type   = "AveragePool"
    pad_const = "0"
    def generate_pool_rhs(self, dim_vars, red_vars, ip_vars, padded_expr):
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
        rhs_expr = "sum({}) / {}".format(
            padded_expr,
            self.generate_funcref(
                count_func,
                dim_vars[-len(red_vars):]))
        return rhs_expr

class GlobalAveragePoolGenerator(AveragePoolGenerator):
    op_type = "GlobalAveragePool"

class MaxPoolGenerator(PoolGenerator):
    op_type   = "MaxPool"
    @property
    def pad_const(self):
        return self.ip0.type.c_min
    def pp_attrs(self):
        PoolGenerator.pp_attrs(self)
        self.has_id_ = hasattr(self, "op1")
    def infer_types(self):
        PoolGenerator.infer_types(self)
        if self.has_id_:
            self.op1.set_type(HalogenType.from_c("int64_t"))

    def infer_shapes(self):
        PoolGenerator.infer_shapes(self)
        if self.has_id_:
            self.op1.set_shape(self.op0.shape)

    def generate_pool_rhs(self, dim_vars, red_vars, ip_vars, padded_expr):
        if self.has_id_:
            maxed = self.generate_funcdecl("maxed")
            self.generate_asn(self.generate_funcref(maxed, dim_vars),
                              "argmax({})".format(padded_expr))
            prod = self.ip0.shape[::-1]
            prod = [int(np.prod(prod[:i])) for i in range(len(prod))]
            if self.storage_order_ == 1:
                prod[:2] = prod[:2][::-1]
            maxed_vars = ["({}*{}+{}[{}]-{})*{}".format(
                dv, st, self.generate_funcref(maxed, dim_vars),
                i-self.n_ign_dims_,
                pad[0], prod) \
                          if rv else "{}*{}".format(dv, prod) \
                          for i, (dv, rv, st, pad, prod) in enumerate(zip(
                                  dim_vars,
                                  [{}]*self.n_ign_dims_ + red_vars,
                                  [{}]*self.n_ign_dims_ + self.strides_,
                                  [{}]*self.n_ign_dims_ + self.pads_,
                                  prod[::-1]))]
            self.generate_asn(self.generate_funcref(self.op1,
                                                    dim_vars),
                              self.generate_cast(self.op1.type,
                                                 "+".join(maxed_vars)))
        rhs = "maximum({})".format(padded_expr)
        return rhs

class GlobalMaxPoolGenerator(MaxPoolGenerator):
    op_type = "GlobalMaxPool"

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
            red_vars[0], dim_vars[1],
            self.op0.shape[1]//self.group_, self.ip0.shape[1]//self.group_)] + \
            ["{}*{}+{}*{}-{}".format(dv, stride, rv, dilation, pad[0]) for \
             dv, rv, pad, stride, dilation in \
             zip(dim_vars[2:], red_vars[1:], self.pads_, self.strides_, self.dilations_)]
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
            bias_expr = self.generate_funcref(self.ip2, [dim_vars[1]])
        else:
            bias_expr = "0"

        w_expr = self.generate_funcref(self.w, w_vars)

        lhs = self.generate_funcref(self.op0, dim_vars)
        rhs = "sum({}*{})+{}".format(padded_expr, w_expr, bias_expr)
        self.generate_asn(lhs, rhs)

class BatchNormGenerator(NodeGenerator):
    op_type = "BatchNormalization"
    attr_fields = {"eps"   : ("epsilon", "f"   , 0),
                   "eps_t" : ("epsilon", "type", HalogenType.from_c("float"))}
    def pp_attrs(self):
        if type(self.eps_t_) != HalogenType:
            self.eps_t_ = HalogenType.from_onnx(self.eps_t_)

        self.x    = self.ip0
        self.s    = self.ip1
        self.bias = self.ip2
        self.mean = self.ip3
        self.var  = self.ip4
    def generate_alg(self, dim_vars):
        lhs    = self.generate_funcref(self.op0, dim_vars)

        s_expr    = self.generate_funcref(self.s, [dim_vars[1]])
        x_expr    = self.generate_funcref(self.x, dim_vars)
        mean_expr = self.generate_funcref(self.mean, [dim_vars[1]])
        var_expr  = self.generate_funcref(self.var, [dim_vars[1]])
        bias_expr = self.generate_funcref(self.bias, [dim_vars[1]])
        eps_expr  = "Expr({})".format(self.eps_)
        rhs = self.generate_cast(
            self.x.type,
            "{}*({}-{})/sqrt({}+{})+{}".format(
                s_expr, x_expr, mean_expr, var_expr, eps_expr, bias_expr))
        self.generate_asn(lhs, rhs)
        
class ConcatGenerator(NodeGenerator):
    op_type = "Concat"
    attr_fields = {"axis" : ("axis", "i", 0)}
    def infer_shapes(self):
        self.op0.set_shape([ip0s + ip1s if i == self.axis_ else ip0s \
                            for i, (ip0s, ip1s) in \
                            enumerate(zip(self.ip0.shape, self.ip1.shape))])

    def generate_alg(self, dim_vars):
        ip0_dim_vars = ["clamp({},0,{})".format(v,
                                                self.ip0.shape[self.axis_] - 1) \
                        if i == self.axis_ else v for i, v \
                        in enumerate(dim_vars)]
        ip1_dim_vars = ["clamp({}-{},0,{})".format(v,
                                                   self.ip0.shape[self.axis_],
                                                   self.ip1.shape[self.axis_] - 1) \
                        if i == self.axis_ else v for i, v \
                        in enumerate(dim_vars)]

        lhs      = self.generate_funcref(self.op0, dim_vars)
        ip0_expr = self.generate_funcref(self.ip0, ip0_dim_vars)
        ip1_expr = self.generate_funcref(self.ip1, ip1_dim_vars)
        rhs = "select({}<{},{},{})".format(
            dim_vars[self.axis_], self.ip0.shape[self.axis_],
            ip0_expr, ip1_expr)
        self.generate_asn(lhs, rhs)

class ConstantGenerator(NodeGenerator):
    op_type = "Constant"
    attr_fields = {"tensor":("value","t",None)}
    def pp_attrs(self):
        self.t_dims_ = tuple(self.tensor_.dims)
        self.t_type_ = HalogenType.from_onnx(self.tensor_.data_type)
        if self.tensor_.raw_data:
            self.t_data_ = np.frombuffer(
                self.tensor_.raw_data,
                count=int(np.prod(self.t_dims_)),
                dtype=self.t_type_.np)
        else:
            self.t_data_ = self.tensor_.float_data
    def infer_shapes(self):
        self.op0.set_shape(self.t_dims_)
    def infer_types(self):
        self.op0.set_type(self.t_type_)
    def generate_alg(self, dim_vars):
        lhs = self.generate_funcref(self.op0, dim_vars)
        if self.op0.is_scalar:
            rhs = self.generate_cast(self.op0.type, self.t_data_[0])
        else:
            self.alg("{0}* {3}_a = new {0}[{1}] {{{2}}};".format(
                self.op0.type.c,
                self.op0.size,
                ",".join(map(str, self.t_data_)),
                self.op0.name))
            self.alg("Buffer<{0}> {1}_b({1}_a, {{{2}}});".format(
                self.op0.type.c,
                self.op0.name,
                ",".join(map(str, self.op0.shape[::-1]))))
            rhs = self.generate_funcref("{0}_b".format(self.op0.name),
                                        dim_vars)
        self.generate_asn(lhs, rhs)

class PadGenerator(NodeGenerator):
    op_type = "Pad"
    attr_fields = {"mode"  :("mode","s","constant"),
                   "pads"  :("pads","ints",None),
                   "const" :("value","f", 0)}
    def pp_attrs(self):
        self.const_      = self.generate_cast(self.ip0.type,
                                              "Expr({})".format(self.const_))
        li               = len(self.pads_) // 2
        self.pads_       = list(zip(self.pads_[:li], self.pads_[li:]))
        self.n_ign_dims_ = self.ip0.dims - len(self.pads_)
    def infer_shapes(self):
        self.op0.set_shape([ips + pad[0] + pad[1] if pad else ips \
                            for pad, ips in zip(
                                    [()] * self.n_ign_dims_ + self.pads_,
                                    self.ip0.shape)])
    def generate_alg(self, dim_vars):
        lhs = "Func {}_pad".format(self.op0.name)
        if self.mode_ == "constant":
            rhs = "constant_exterior({},{},{{{}}})".format(
                self.ip0.name, self.const_,
                ",".join(["{{0,{}}}".format(ips) \
                          for ips in self.ip0.shape[::-1]]))
        elif self.mode_ == "edge":
            rhs = "repeat_edge({},{{{}}})".format(
                self.ip0.name,
                ",".join(["{{0,{}}}".format(ips) \
                          for ips in self.ip0.shape[::-1]]))
        elif self.mode_ == "reflect":
            rhs = "mirror_interior({},{{{}}})".format(
                self.ip0.name,
                ",".join(["{{0,{}}}".format(ips) if ips > 1 else "{Expr(),Expr()}"\
                          for ips in self.ip0.shape[::-1]]))
        self.generate_asn(lhs, rhs)

        ip_vars = ["{}-{}".format(dv, pad[0]) if pad else dv \
                   for dv, pad in zip(dim_vars, [{}]*self.n_ign_dims_+self.pads_)]
        
        op_expr = self.generate_funcref(self.op0, dim_vars)
        ip_expr = self.generate_funcref("{}_pad".format(self.op0.name),
                                        ip_vars)
        self.generate_asn(op_expr, ip_expr)

class ConvTGenerator(NodeGenerator):
    op_type     = "ConvTranspose"
    attr_fields = {"kernel_shape":("kernel_shape"  , "ints", None),
                   "pads"        :("pads"          , "ints", None),
                   "strides"     :("strides"       , "ints", None),
                   "dilations"   :("dilations"     , "ints", None),
                   "op_shape"    :("output_shape"  , "ints", None),
                   "op_pads"     :("output_padding", "ints", None),
                   "auto_pad"    :("auto_pad"      , "s"   , None)}
    def pp_attrs(self):
        if self.pads_:
            li = len(self.pads_) // 2
            self.pads_     = list(zip(self.pads_[:li], self.pads_[:li]))
        self.w             = self.ip1
        self.bias          = self.ip2 if hasattr(self, "ip2") else None
        self.strides_      = self.strides_ or [1] * (self.w.dims - 2)
        self.dilations_    = self.dilations_ or [1] * (self.w.dims - 2)
        self.kernel_shape_ = self.kernel_shape_ or self.w.shape[2:]
        self.pads_         = self.pads_ or [(0, 0)] * len(self.kernel_shape_)
        self.op_pads_      = self.op_pads_ or [0] * len(self.kernel_shape_)
        if self.op_shape_:
            if len(self.op_shape_) < self.ip0.dims:
                self.op_shape_ = [self.ip0.shape[0], self.w.shape[1]] + self.op_shape_
            total_padding = [stride*(ops-1)+op_pad+ks-ips \
                             for (ips, ops, op_pad, ks, stride) in \
                             zip(self.ip0.shape[2:],
                                 self.op_shape_[2:],
                                 self.op_pads_,
                                 self.kernel_shape_,
                                 self.strides_)]
            if self.auto_pad_:
                if self.auto_pad_ != "SAME_UPPER":
                    self.pads_ = [(tp//2, tp-tp//2) for tp in total_padding]
                else:
                    self.pads_ = [(tp-tp//2, tp//2) for tp in total_padding]
        else:
            self.op_shape_ = [self.ip0.shape[0], self.w.shape[1]] + \
                             [stride*(ips-1)+op_pad+ks-pad[0]-pad[1] \
                              for (ips, op_pad, ks, stride, pad) in
                              zip(self.ip0.shape[2:],
                                  self.op_pads_,
                                  self.kernel_shape_,
                                  self.strides_,
                                  self.pads_)]
    def infer_shapes(self):
        self.op0.set_shape(self.op_shape_)
    def generate_alg(self, dim_vars):
        red_vars = self.generate_rdom("r", [(0, i) for i in self.ip0.shape[1:]])
        pad_w = self.generate_padded("pad_w", self.w, 0,
                                     [("Expr()","Expr()")]*2 + \
                                     [(0, s) for s in self.w.shape[2:]])
        pad_i = self.generate_padded("pad_i", self.ip0, 0,
                                     [("Expr()","Expr()")]*2 + 
                                     [(0, s) for s in self.ip0.shape[2:]])

        dilated = self.generate_funcdecl("dilated")
        dilated_lhs = self.generate_funcref(dilated, dim_vars)
        dilated_cond = ["(({}%{})==0)".format(dv, dil) \
                        for dil, dv in zip(self.dilations_, dim_vars[2:])]
        dilated_w_vars = ["cast<int>(floor({}/{}))".format(dv, dil) \
                          for dil, dv in zip(self.dilations_, dim_vars[2:])]
        dilated_rhs = "select({}, {}, 0)".format(
            "&&".join(dilated_cond),
            self.generate_funcref(pad_w, dim_vars[:2] + dilated_w_vars))
        self.generate_asn(dilated_lhs, dilated_rhs)

        ip_vars = [dim_vars[0], red_vars[0]] + \
                  ["cast<int>(floor(({0}-{4}*{5}+{2})/{1}))".format(dv, st, pad[0], op_pad, rv, dil) \
                   for dv, st, pad, op_pad, rv, dil \
                   in zip(dim_vars[2:],
                          self.strides_,
                          self.pads_,
                          self.op_pads_,
                          red_vars[1:],
                          self.dilations_)]
        w_vars = [red_vars[0], dim_vars[1]] + \
                 ["{}*{}".format(rv, dil) for \
                  dv, rv, pad, dil, stride in \
                  zip(dim_vars[2:],
                      red_vars[1:],
                      self.pads_,
                      self.dilations_,
                      self.strides_)]
        sel_expr = ["((({0}-{2}*{3}+{5})%{1})==0)".format(dv, st, rv, dil, op_pad, pad[0]) \
                    for dv, rv, st, dil, op_pad, pad in \
                    zip(dim_vars[2:],
                        red_vars[1:],
                        self.strides_,
                        self.dilations_,
                        self.op_pads_,
                        self.pads_)]
        if self.bias:
            bias_expr = "+{}".format(self.generate_funcref(self.bias,
                                                           [dim_vars[1]]))
        else:
            bias_expr = ""
        lhs = self.generate_funcref(self.op0, dim_vars)
        rhs = "sum(select({},{},0)*{}){}".format(
            "&&".join(sel_expr),
            self.generate_funcref(pad_i, ip_vars),
            self.generate_funcref(dilated, w_vars),
            bias_expr)
        self.generate_asn(lhs, rhs)
        
class DToSGenerator(NodeGenerator):
    op_type = "DepthToSpace"
    attr_fields = {"blocksize":("blocksize", "i", None)}
    def infer_shapes(self):
        self.op0.set_shape([self.ip0.shape[0],
                            self.ip0.shape[1]//(self.blocksize_**2),
                            self.ip0.shape[2]*self.blocksize_,
                            self.ip0.shape[3]*self.blocksize_])
    def generate_alg(self, dim_vars):
        ip_vars = [dim_vars[0],
                   "{}+({}%{})*{}+({}%{})*{}".format(
                       dim_vars[1],
                       dim_vars[3], self.blocksize_,
                       self.ip0.shape[1] // (self.blocksize_**2),
                       dim_vars[2], self.blocksize_,
                       self.ip0.shape[1] // self.blocksize_),
                   "cast<int>({}/{})".format(
                       dim_vars[2],
                       self.blocksize_),
                   "cast<int>({}/{})".format(
                       dim_vars[3],
                       self.blocksize_)]
        self.generate_asn(self.generate_funcref(self.op0,
                                                dim_vars),
                          self.generate_funcref(self.ip0,
                                                ip_vars))

class FlattenGenerator(NodeGenerator):
    op_type = "Flatten"
    attr_fields = {"axis":("axis", "i", 1)}
    def infer_shapes(self):
        if self.axis_ == 0:
            self.op0.set_shape([1, self.ip0.size])
        else:
            self.op0.set_shape([int(np.prod(self.ip0.shape[:self.axis_])),
                                int(np.prod(self.ip0.shape[self.axis_:]))])
    def generate_alg(self, dim_vars):
        if self.axis_ == 0:
            prevs = self.ip0.shape[1:] + [1]
            prods = [np.prod(prevs[i:]) for i in range(len(prevs))]
            ip_vars = ["cast<int>(floor({}/{}))%{}".format(dim_vars[1], prod, ips) \
                       for ips, prod in zip(
                               self.ip0.shape,
                               prods)]
        else:
            pprevs = self.ip0.shape[self.axis_:] + [1]
            pprods = [np.prod(pprevs[i:]) for i in range(len(pprevs))]
            fprevs = self.ip0.shape[:self.axis_] + [1]
            fprods = [np.prod(fprevs[i:]) for i in range(len(fprevs))]
            ip_vars = ["cast<int>(floor({}/{}))%{}".format(dim_vars[0], prod, ips) \
                       for ips, prod in zip(
                               fprevs,
                               fprods[1:])] \
                    + ["cast<int>(floor({}/{}))%{}".format(dim_vars[1], prod, ips) \
                       for ips, prod in zip(
                          pprevs,
                          pprods[1:])]
        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          self.generate_funcref(self.ip0, ip_vars))

class GatherGenerator(NodeGenerator):
    op_type = "Gather"
    attr_fields = {"axis":("axis","i",0)}
    def infer_shapes(self):
        self.op0.set_shape(self.ip0.shape[:self.axis_] \
                           + self.ip1.shape \
                           + self.ip0.shape[self.axis_+1:])
    def generate_alg(self, dim_vars):
        id_vars = dim_vars[self.axis_:self.axis_+self.ip1.dims]
        ip_vars = dim_vars[:self.axis_] \
                  + ["clamp(cast<int>({}),0,{})".format(
                      self.generate_funcref(self.ip1, id_vars),
                      self.ip0.shape[self.axis_]-1)] \
                  + dim_vars[self.ip1.dims+self.axis_:]
        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          self.generate_funcref(self.ip0, ip_vars))

class GemmGenerator(NodeGenerator):
    op_type = "Gemm"
    attr_fields = {"alpha" :("alpha","f",1),
                   "beta"  :("beta","f",1),
                   "transA":("transA","i",0),
                   "transB":("transB","i",0)}
    def pp_attrs(self):
        self.A = self.ip0
        self.B = self.ip1
        self.C = self.ip2
        self.Y = self.op0
        if self.transA_:
            self.K_, self.M_ = self.A.shape
        else:
            self.M_, self.K_ = self.A.shape
        if self.transB_:
            self.N_ = self.B.shape[0]
        else:
            self.N_ = self.B.shape[1]
    def infer_shapes(self):
        self.Y.set_shape([self.M_, self.N_])

    def generate_alg(self, dim_vars):
        alpha = self.generate_cast(self.C.type, "Expr({})".format(self.alpha_))
        beta  = self.generate_cast(self.C.type, "Expr({})".format(self.beta_))
        red_var = self.generate_rdom("r", [(0, self.K_)])[0]

        norm_A = self.generate_funcdecl("norm_A")
        norm_B = self.generate_funcdecl("norm_B")
        norm_C = self.generate_funcdecl("norm_C")
        if self.transA_:
            self.generate_asn(self.generate_funcref(norm_A, dim_vars[:2]),
                              self.generate_funcref(self.A, dim_vars[:2][::-1]))
        else:
            self.generate_asn(norm_A, self.A.name)
        if self.transB_:
            self.generate_asn(self.generate_funcref(norm_B, dim_vars[:2]),
                              self.generate_funcref(self.B, dim_vars[:2][::-1]))
        else:
            self.generate_asn(norm_B, self.B.name)
        self.generate_asn(self.generate_funcref(norm_C, dim_vars),
                          self.generate_funcref(self.C, [dv if cs > 1 else "0" \
                                                         for dv, cs \
                                                         in zip(dim_vars[::-1],
                                                                self.C.shape[::-1])][::-1]))
        self.generate_asn(self.generate_funcref(self.Y, dim_vars),
                          "{}*{}+{}*sum({}*{})".format(
                              beta,
                              self.generate_funcref(norm_C, dim_vars),
                              alpha,
                              self.generate_funcref(norm_A, [dim_vars[0], red_var]),
                              self.generate_funcref(norm_B, [red_var, dim_vars[1]])))

class FeaturemaxGenerator(NodeGenerator):
    attr_fields = {"axis":("axis", "i", 1)}
    def generate_alg(self, dim_vars):
        red_vars = self.generate_rdom("r",
                                      [(0, s) for s in self.ip0.shape[self.axis_:]])
        ip_vars = dim_vars[:self.axis_] + red_vars

        lhs = self.generate_funcref(self.op0, dim_vars)
        rhs = self.generate_rhs(dim_vars, ip_vars, red_vars)
        self.generate_asn(lhs, rhs)

class HardmaxGenerator(FeaturemaxGenerator):
    op_type = "Hardmax"
    def generate_rhs(self, dim_vars, ip_vars, red_vars):
        self.alg("Tuple {}_am = argmax({});".format(
            self.op0.name,
            self.generate_funcref(self.ip0, ip_vars)))
        return self.generate_cast(self.op0.type,
                                  "&&".join(
                                      ["({}_am[{}]=={})".format(self.op0.name,
                                                                i,
                                                                dv)
                                       for i, dv in enumerate(dim_vars[self.axis_:])]))

class LogSoftmaxGenerator(FeaturemaxGenerator):
    op_type = "LogSoftmax"
    def generate_rhs(self, dim_vars, ip_vars, red_vars):
        norm_ip = self.generate_funcdecl("norm_ip")
        self.generate_asn(self.generate_funcref(norm_ip, dim_vars),
                          "{}-maximum({})".format(
                              self.generate_funcref(self.ip0, dim_vars),
                              self.generate_funcref(self.ip0, ip_vars)))
        return "({}-log(sum(exp({}))))".format(
            self.generate_funcref(norm_ip, dim_vars),
            self.generate_funcref(norm_ip, ip_vars))

class SoftmaxGenerator(FeaturemaxGenerator):
    op_type = "Softmax"
    def generate_rhs(self, dim_vars, ip_vars, red_vars):
        max_x = self.generate_funcdecl("max_x")
        self.generate_asn(self.generate_funcref(max_x, dim_vars),
                          "maximum({})".format(self.generate_funcref(
                              self.ip0, ip_vars)))
        exp_x = self.generate_funcdecl("exp_x")
        self.generate_asn(self.generate_funcref(exp_x, dim_vars),
                          "exp({}-{})".format(
                              self.generate_funcref(self.ip0, dim_vars),
                              self.generate_funcref(max_x, dim_vars)))
        return "{}/sum({})".format(
            self.generate_funcref(exp_x, dim_vars),
            self.generate_funcref(exp_x, ip_vars))

class InstanceNormGenerator(NodeGenerator):
    op_type = "InstanceNormalization"
    attr_fields = {"eps": ("epsilon", "f", 1e-5)}
    def generate_alg(self, dim_vars):
        red_vars = self.generate_rdom("r", [(0,s) for s in self.op0.shape[2:]])
        eps = self.generate_cast(self.op0.type, "Expr({})".format(self.eps_))

        mean_f = self.generate_funcdecl("mean_f")
        self.generate_asn(self.generate_funcref(mean_f, dim_vars[:2]),
                          "sum({})/{}".format(
                              self.generate_funcref(self.ip0, dim_vars[:2] + red_vars),
                              np.prod(self.ip0.shape[2:])))

        var_f = self.generate_funcdecl("var_f")
        self.generate_asn(self.generate_funcref(var_f, dim_vars[:2]),
                          "(sum(pow({},2))/{} - pow({},2))".format(
                              self.generate_funcref(self.ip0, dim_vars[:2] + red_vars),
                              np.prod(self.ip0.shape[2:]),
                              self.generate_funcref(mean_f, dim_vars[:2])))

        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          "({}*({}-{})/(sqrt({}+{}))+{})".format(
                              self.generate_funcref(self.ip1, [dim_vars[1]]),
                              self.generate_funcref(self.ip0, dim_vars),
                              self.generate_funcref(mean_f, dim_vars[:2]),
                              self.generate_funcref(var_f, dim_vars[:2]),
                              eps,
                              self.generate_funcref(self.ip2, [dim_vars[1]])))

class LRNGenerator(NodeGenerator):
    op_type = "LRN"
    attr_fields = {"alpha":("alpha", "f", 0.0001),
                   "beta" :("beta" , "f", 0.75),
                   "bias" :("bias" , "f", 1.0),
                   "size" :("size" , "i", None)}
    def generate_alg(self, dim_vars):
        red_var = self.generate_rdom("r", [(-floor((self.size_-1)/2),
                                            self.size_)])[0]
        padded = self.generate_padded("pad",
                                      self.ip0,
                                      0,
                                      [("Expr()", "Expr()") if i != 1 else \
                                       (0, self.ip0.shape[1]) for i in \
                                       range(self.ip0.dims)])
        sq_sum = self.generate_funcdecl("sq_sum")
        self.generate_asn(self.generate_funcref(sq_sum, dim_vars),
                          "sum(pow({},2))".format(
                              self.generate_funcref(
                                  padded,
                                  [dim_vars[0],
                                   "{}+{}".format(red_var, dim_vars[1])] \
                                   + dim_vars[2:])))

        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          "{}/pow({}+({}/{})*{},{})".format(
                              self.generate_funcref(self.ip0, dim_vars),
                              self.generate_cast(self.op0.type, self.bias_),
                              self.generate_cast(self.op0.type, self.alpha_),
                              self.generate_cast(self.op0.type, self.size_),
                              self.generate_funcref(sq_sum, dim_vars),
                              self.generate_cast(self.op0.type, self.beta_)))

class MatMulGenerator(NodeGenerator):
    op_type = "MatMul"
    def infer_shapes(self):
        if self.ip0.dims == self.ip1.dims == 2:
            self._case = 0
            op_shape = [self.ip0.shape[0], self.ip1.shape[1]]
        elif self.ip0.dims > 2 and self.ip1.dims == 2:
            self._case = 1
            op_shape = self.ip0.shape[:-1] + [self.ip1.shape[1]]
        elif self.ip0.dims == 2 and self.ip1.dims > 2:
            self._case = 2
            op_shape = self.ip1.shape[:-1] + [self.ip0.shape[0], self.ip1.shape[-1]]
        elif self.ip0.dims > 2 and self.ip1.dims > 2:
            self._case = 3
            op_shape = self.ip1.shape[:-2] + [self.ip0.shape[-2], self.ip1.shape[-1]]
        self.op0.set_shape(op_shape)
    def generate_alg(self, dim_vars):
        K = self.ip0.shape[-1]
        red_var = self.generate_rdom("r", [(0, K)])[0]
        if self._case == 0:
            a_vars = [dim_vars[0], red_var]
            b_vars = [red_var, dim_vars[1]]
        elif self._case == 1:
            a_vars = dim_vars[:-1] + [red_var]
            b_vars = [red_var, dim_vars[-1]]
        elif self._case == 2:
            a_vars = [dim_vars[-2], red_var]
            b_vars = dim_vars[:-2] + [red_var, dim_vars[-1]]
        elif self._case == 3:
            a_vars = dim_vars[:-1] + [red_var]
            b_vars = dim_vars[:-2] + [red_var, dim_vars[-1]]

        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          "sum({}*{})".format(
                              self.generate_funcref(self.ip0, a_vars),
                              self.generate_funcref(self.ip1, b_vars)))


class VarGenerator(NodeGenerator):
    def infer_shape(self):
        if len(self.ips) == 1:
            self.op0.set_shape(self.ip0.shape)
        else:
            dims = max([len(ip.shape) for ip in self.ips])
            op_shape = [1] * dims
            for ip in self.ips:
                op_shape = [op_shape[-i] if i > ip.dims else \
                            max(ip.shape[-i], op.shape[-i]) for \
                            i in range(1, dims+1)][::-1]
            self.op0.set_shape(op_shape)
    def generate_alg(self, dim_vars):
        lhs = self.generate_funcref(self.op0, dim_vars)
        if len(self.ips) == 1:
            rhs = self.generate_funcref(self.ip0, dim_vars)
        else:
            ip_vars = []
            for ip in self.ips:
                ip_vars.append([(dv if dim > 1 else "0") for dim, dv in\
                                zip(ip.shape, dim_vars[-ip.dims:])])
            rhs = self.generate_rhs(dim_vars, ip_vars)
        self.generate_asn(lhs, rhs)

class MinMaxGenerator(VarGenerator):
    def generate_rhs(self, dim_vars, ip_vars):
        return "{}({})".format(
            self.op_type.lower(),
            ",".join([self.generate_funcref(ip, ipv) \
                      for ip, ipv in zip(self.ips, ip_vars)]))
class MaxGenerator(MinMaxGenerator):
    op_type = "Max"
class MinGenerator(MinMaxGenerator):
    op_type = "Min"

class MeanGenerator(VarGenerator):
    op_type = "Mean"
    def generate_rhs(self, dim_vars, ip_vars):
        return "({})/{}".format(
            "+".join([self.generate_funcref(ip, ipv) for\
                      ip, ipv in zip(self.ips, ip_vars)]),
            self.generate_cast(self.op0.type, len(self.ips)))

class SumGenerator(VarGenerator):
    op_type = "Sum"
    def generate_rhs(self, dim_vars, ip_vars):
        return "{}".format(
            "+".join([self.generate_funcref(ip, ipv) for \
                      ip, ipv in zip(self.ips, ip_vars)]))

class PReluGenerator(NodeGenerator):
    op_type = "PRelu"
    def generate_alg(self, dim_vars):
        i = 0
        while self.ip1.shape[-1] > 1 and self.ip1.shape[-1] != self.ip0.shape[::-1][i]:
            i += 1
        ip1_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in\
                        zip(self.ip1.shape[::-1], dim_vars[::-1][i:])][::-1]

        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          "select({0}<0,{0}*{1},{0})".format(
                              self.generate_funcref(self.ip0, dim_vars),
                              self.generate_funcref(self.ip1, ip1_dim_vars)))

class RedlGenerator(NodeGenerator):
    attr_fields = {"keepdims":("keepdims", "i"   , 1),
                   "axes"    :("axes"    , "ints", None)}
    def pp_attrs(self):
        self.axes_ = self.axes_ or list(range(self.ip0.dims))
    def infer_shapes(self):
        if self.keepdims_:
            op_shape = [s if i not in self.axes_ else 1 \
                        for i, s in enumerate(self.ip0.shape)]
        else:
            op_shape = [s for i, s in enumerate(self.ip0.shape) if i not in self.axes_]
        self.op0.set_shape(op_shape)
    def generate_alg(self, dim_vars):
        red_vars = list(zip(self.axes_,
                            self.generate_rdom("r",
                                               [(0,self.ip0.shape[i]) \
                                                for i in self.axes_])))

        if self.keepdims_:
            zd_vars = list(zip([i for i in range(len(self.ip0.shape)) \
                                if i not in self.axes_],
                               [dv for i, dv in enumerate(dim_vars) \
                                if i not in self.axes_]))
        else:
            zd_vars = list(zip([i for i in range(len(self.ip0.shape)) \
                                if i not in self.axes_],
                               dim_vars))
        ip_vars = [None] * self.ip0.dims
        for i, rv in red_vars:
            ip_vars[i] = rv
        for i, dv in zd_vars:
            ip_vars[i] = dv

        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          self.expr.format(self.generate_funcref(self.ip0,
                                                                 ip_vars)))

class ReduceL1Generator(RedlGenerator):
    op_type = "ReduceL1"
    expr    = "sum(abs({}))"


class ReduceL2Generator(RedlGenerator):
    op_type = "ReduceL2"
    expr    = "sqrt(sum(pow({},2)))"

class ReduceLogSum(RedlGenerator):
    op_type = "ReduceLogSum"
    expr    = "log(sum({}))"

class ReduceLogSumExp(RedlGenerator):
    op_type = "ReduceLogSumExp"
    expr    = "log(sum(exp({})))"

class ReduceMax(RedlGenerator):
    op_type = "ReduceMax"
    expr    = "maximum({})"

class ReduceMean(RedlGenerator):
    op_type = "ReduceMean"
    @property
    def expr(self):
        return "sum({{}})/{}".format(
            np.prod([self.ip0.shape[i] for i in self.axes_]))

class ReduceMin(RedlGenerator):
    op_type = "ReduceMin"
    expr    = "minimum({})"

class ReduceProd(RedlGenerator):
    op_type = "ReduceProd"
    expr    = "product({})"

class ReduceSum(RedlGenerator):
    op_type = "ReduceSum"
    expr    = "sum({})"

class ReduceSumSquare(RedlGenerator):
    op_type = "ReduceSumSquare"
    expr    = "sum(pow({},2))"

class ShapeGenerator(NodeGenerator):
    op_type = "Shape"
    def infer_shapes(self):
        self.op0.set_shape([self.ip0.dims])
    def infer_types(self):
        self.op0.set_type(HalogenType.from_c("int64_t"))
    def generate_alg(self, dim_vars):
        self.alg("int64_t* {}_c = new int64_t[{}] {{{}}};".format(
            self.op0.name,
            self.ip0.dims,
            ",".join(map(str,self.ip0.shape))))
        self.alg("Buffer<int64_t> {0}_b({0}_c, {{{1}}});".format(
            self.op0.name,
            self.ip0.dims))
        self.alg("Func {0}_fb({0}_b);".format(
            self.op0.name))
        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          self.generate_funcref("{}_fb".format(self.op0.name),
                                                dim_vars))

class SizeGenerator(NodeGenerator):
    op_type = "Size"
    def infer_shapes(self):
        self.op0.set_shape(())
    def infer_types(self):
        self.op0.set_type(HalogenType.from_c("int64_t"))
    def generate_alg(self, dim_vars):
        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          self.generate_cast(self.op0.type, self.ip0.size))

class SliceGenerator(NodeGenerator):
    op_type = "Slice"
    attr_fields = {"axes":("axes","ints",None),
                   "ends":("ends","ints",None),
                   "starts":("starts","ints",None)}
    def pp_attrs(self):
        self.axes_ = self.axes_ or list(range(len(self.starts_)))
    def infer_shapes(self):
        op_shape = self.ip0.shape
        self.st_dict_ = {}
        for (i, s, e) in zip(self.axes_, self.starts_, self.ends_):
            s = max(0, min(self.ip0.shape[i], s))
            if e < 0:
                e = self.ip0.shape[i] + e
            e = min(self.ip0.shape[i], e)
            op_shape[i] = e - s
            self.st_dict_[i] = s
        self.op0.set_shape(op_shape)
    def generate_alg(self, dim_vars):
        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          self.generate_funcref(self.ip0,
                                                ["{}+{}".format(dv,
                                                                self.st_dict_[i]) \
                                                 if i in self.st_dict_ else dv \
                                                 for i, dv in \
                                                 enumerate(dim_vars)]))

class SplitGenerator(NodeGenerator):
    op_type = "Split"
    attr_fields = {"axis":("axis","i",0),
                   "split":("split","ints",None)}
    def pp_attrs(self):
        self.split_ = self.split_ or [int(self.ip0.shape[self.axis_]//len(self.ops))] * len(self.ops)

    def infer_types(self):
        for op in self.ops:
            op.set_type(self.ip0.type)
    def infer_shapes(self):
        for op, s in zip(self.ops, self.split_):
            op_shape = self.ip0.shape
            op_shape[self.axis_] = s
            op.set_shape(op_shape)
    def generate_alg(self, dim_vars):
        s_sum = 0
        for op, s in zip(self.ops, self.split_):
            ip_vars = list(dim_vars)
            ip_vars[self.axis_] += "+{}".format(s_sum)
            self.generate_asn(self.generate_funcref(op, dim_vars),
                              self.generate_funcref(self.ip0, ip_vars))
            s_sum += s
        
class SqueezeGenerator(NodeGenerator):
    op_type = "Squeeze"
    attr_fields = {"axes":("axes", "ints", None)}
    def infer_shapes(self):
        self.op0.set_shape([s for i, s in enumerate(self.ip0.shape) \
                            if i not in self.axes_])
    def generate_alg(self, dim_vars):
        ip_vars = ["0"] * len(self.ip0.shape)
        for i, dv in zip([i for i in range(len(self.ip0.shape)) \
                          if i not in self.axes_],
                         dim_vars):
            ip_vars[i] = dv
        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          self.generate_funcref(self.ip0, ip_vars))

class TransposeGenerator(NodeGenerator):
    op_type = "Transpose"
    attr_fields = {"perms":("perm","ints",None)}
    def pp_attrs(self):
        self.perms_ = self.perms_ or list(range(self.ip0.dims))[::-1]
    def infer_shapes(self):
        self.op0.set_shape([self.ip0.shape[i] for i in self.perms_])
    def generate_alg(self, dim_vars):
        self.generate_asn(self.generate_funcref(self.op0,
                                                [dim_vars[i] for i in self.perms_]),
                          self.generate_funcref(self.ip0, dim_vars))

class UnsqueezeGenerator(NodeGenerator):
    op_type = "Unsqueeze"
    attr_fields = {"axes":("axes","ints",None)}
    def infer_shapes(self):
        op_shape = self.ip0.shape
        self.orig_s_ = [1] * self.ip0.dims
        for i in self.axes_:
            op_shape.insert(i, 1)
            self.orig_s_.insert(i, 0)
        self.op0.set_shape(op_shape)
    def generate_alg(self, dim_vars):
        ip_vars = [dv for i, dv in enumerate(dim_vars) if self.orig_s_[i]]
        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          self.generate_funcref(self.ip0, ip_vars))
            
class ReshapeGenerator(NodeGenerator):
    op_type = "Reshape"
    def infer_shapes(self):
        self.op0.set_shape(tuple(map(int, self.ip1.data)))
    def generate_alg(self, dim_vars):
        prevs = self.ip0.shape[1:] + [1]
        prods = [np.prod(prevs[i:]) for i in range(len(prevs))]
        ip_vars = ["cast<int>(floor({}/{}))%{}".format(dim_vars[0],
                                                       prod,
                                                       ips) \
                   for ips, prod in \
                   zip(self.ip0.shape, prods)]

        flattened = self.generate_funcdecl("flattened")
        self.generate_asn(self.generate_funcref(flattened, [dim_vars[0]]),
                          self.generate_funcref(self.ip0, ip_vars))
        prevs = self.op0.shape[1:] + [1]
        prods = [np.prod(prevs[i:]) for i in range(len(prevs))]

        fl_vars = ["({}*{})".format(dv, p) for dv, p in zip(dim_vars, prods)]
        self.generate_asn(self.generate_funcref(self.op0, dim_vars),
                          self.generate_funcref(flattened, ["+".join(fl_vars)]))
