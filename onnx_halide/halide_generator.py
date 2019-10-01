from .base_generator import BaseGraphVisitor, BaseNodeVisitor
from .types import VI
import numpy as np
import os
from os.path import join, dirname
from . import __path__
from math import floor, ceil
from .environment_link import Environment

from onnx.onnx_ml_pb2 import NodeProto, TypeProto
from typing import Dict, List, Set, Tuple, Union
class HalideGraphVisitor(BaseGraphVisitor):
    pass

def JOIN_VARS(strs: List[str]) -> str:
    return ','.join(strs[::-1])


class HalideNodeVisitor(BaseNodeVisitor):
    attr_fields = {}
    @property
    def n_dim_vars(self) -> int:
        return len(VI(self.value_info[self.outputs[0]]).shape)

    def __init__(self, **kwargs) -> None:
        BaseNodeVisitor.__init__(self, **kwargs)
        halide_runtime = join(dirname(__path__[0]), "runtime/HalideRuntime.o")
        BaseGraphVisitor.register_runtime({halide_runtime}, set())


    def generate_alg(self, dim_vars):
        pass

    def visit(self, node: NodeProto, value_info: Dict[str, TypeProto]) -> Tuple[List[str], Set[str], Set[str]]:
        BaseNodeVisitor.visit(self, node, value_info)


        dim_vars = ["d{}".format(i) for i in range(self.n_dim_vars)]

        gen_name = "HalideNode_{}".format(self.outputs[0])

        input_decls = '\n'.join(["Input<Buffer<{0}>> v_{1}{{\"v_{1}\", {2}}};".format(
            VI(value_info[i]).t.c,
            i,
            len(VI(value_info[i]).shape),
            VI(value_info[i]).shape) for i in self.inputs if i])

        output_decls = '\n'.join(["Output<Buffer<{0}>> v_{1}{{\"v_{1}\", {2}}};".format(
            VI(value_info[i]).t.c,
            i,
            len(VI(value_info[i]).shape),
            VI(value_info[i]).shape) for i in self.outputs])

        dim_var_decls = '\n'.join(["Var {};".format(d) for d in dim_vars])


        alg = self.generate_alg(dim_vars)

        src = """
#include "Halide.h"
#include <stdint.h>
#include <cfloat>
#include <limits.h>

using namespace Halide;
using namespace BoundaryConditions;
namespace {{
class {0} : public Generator<{0}> {{
public:
{1}

{2}

{3}

void generate() {{

{4}

}};
}};
}}
HALIDE_REGISTER_GENERATOR({0}, {0})
"""
        src = src.format(gen_name, input_decls, output_decls, dim_var_decls,
                         '\n'.join(alg))

        Environment.compile_kernel(src, gen_name, self.temp_dir)

        code = []
        haargs   = []

        for i in self.inputs + self.outputs:
            if not i:
                continue
            vi = VI(value_info[i])
            ctype = vi.t.c
            shape = vi.shape

            code.append("  Halide::Runtime::Buffer<{0}> {1}_buf(v_{1}, {{{2}}});".format(
                ctype,
                i,
                JOIN_VARS([str(i) for i in shape])))
            haargs.append("{}_buf".format(i))

        code.append("  {}({});".format(gen_name, ','.join(haargs)))

        code = ["{"] + code + ["};"]
        objects = {join(self.temp_dir, "{}.o".format(gen_name))}
        headers = set(["\"{}\"".format(n) for n in
                       [join(self.temp_dir, "{}.h".format(gen_name)),
                        join(self.install_dir, "include/HalideBuffer.h"),
                        join(self.install_dir, "include/HalideRuntime.h")]])
        return code, objects, headers



    def generate_funcref(self, func: str, dim_vars: List[str]) -> str:
        return "{}({})".format(func, JOIN_VARS(dim_vars))

    def generate_assign(self, lhs: str, rhs: str) -> str:
        return "{} = {};".format(lhs, rhs)

    def generate_rdom(self, name: str, ranges: List[Tuple[int, int]]) -> Tuple[str, List[str]]:
        rdom_name = "{}_{}".format(self.outputs[0], name)
        code = "RDom {}({});".format(rdom_name,
                                      ','.join(["{},{}".format(a,b) \
                                                for a, b in ranges]))
        return code, ["{}[{}]".format(rdom_name, i) for i in range(len(ranges))]

    def generate_cast(self, type_: str, expr: float) -> str:
        return "cast<{}>(Expr({}))".format(type_, expr)

    def generate_funcdecl(self, name: str) -> Tuple[str, str]:
        name = "{}_{}".format(self.outputs[0], name)
        return "Func {};".format(name), name

    def generate_padded(self, name: str, ip: str, pad_const: int, pad_doms: List[Union[Tuple[str, str], Tuple[int, int]]]) -> Tuple[str, str]:
        name = "{}_{}".format(self.outputs[0], name)
        return "Func {} = {};".format(
            name,
            "constant_exterior({}, {}, {{{}}})".format(
                ip,
                pad_const,
                JOIN_VARS(["{{{},{}}}".format(a, b) for a, b in pad_doms]))), name


class HalideUnaryVisitor(HalideNodeVisitor):
    def generate_alg(self, dim_vars):
        ip0_expr  = self.generate_funcref("v_" + self.inputs[0], dim_vars)
        unop_expr = self.expr.format(ip0_expr)
        op0_expr  = self.generate_funcref("v_" + self.outputs[0], dim_vars)
        return [self.generate_assign(op0_expr, unop_expr)]


class HalideAbsVisitor(HalideUnaryVisitor):
    op_type = "Abs"
    expr    = "abs({})"
HalideGraphVisitor.register(HalideAbsVisitor)

class HalideAcosVisitor(HalideUnaryVisitor):
    op_type = "Acos"
    expr    = "acos({})"
HalideGraphVisitor.register(HalideAcosVisitor)

class HalideAcoshVisitor(HalideUnaryVisitor):
    op_type = "Acosh"
    expr    = "acosh({})"
HalideGraphVisitor.register(HalideAcoshVisitor)

class HalideAsinVisitor(HalideUnaryVisitor):
    op_type = "Asin"
    expr    = "asin({})"
HalideGraphVisitor.register(HalideAsinVisitor)

class HalideAsinhVisitor(HalideUnaryVisitor):
    op_type = "Asinh"
    expr    = "asinh({})"
HalideGraphVisitor.register(HalideAsinhVisitor)

class HalideAtanVisitor(HalideUnaryVisitor):
    op_type = "Atan"
    expr    = "atan({})"
HalideGraphVisitor.register(HalideAtanVisitor)

class HalideAtanhVisitor(HalideUnaryVisitor):
    op_type = "Atanh"
    expr    = "atanh({})"
HalideGraphVisitor.register(HalideAtanhVisitor)

class HalideCosVisitor(HalideUnaryVisitor):
    op_type = "Cos"
    expr    = "cos({})"
HalideGraphVisitor.register(HalideCosVisitor)

class HalideCoshVisitor(HalideUnaryVisitor):
    op_type = "Cosh"
    expr    = "cosh({})"
HalideGraphVisitor.register(HalideCoshVisitor)

class HalideCastVisitor(HalideUnaryVisitor):
    op_type = "Cast"
    attr_fields = {"to":("to", "i", None)}
    @property
    def expr(self):
        return self.generate_cast(VI(self.value_info[self.outputs[0]]).t.c, "{}")
HalideGraphVisitor.register(HalideCastVisitor)

class HalideCeilVisitor(HalideUnaryVisitor):
    op_type = "Ceil"
    expr    = "ceil({})"
HalideGraphVisitor.register(HalideCeilVisitor)

class HalideClipVisitor(HalideUnaryVisitor):
    op_type = "Clip"

    @property
    def expr(self):
        min_v = None
        max_v = None
        n = len(self.inputs)
        if n > 1 and self.inputs[1]:
            min_v = "v_{}()".format(self.inputs[1])
        if n > 2 and self.inputs[2]:
            max_v = "v_{}()".format(self.inputs[2])
        min_v = min_v or "Expr({})".format(min_v or VI(self.value_info[self.inputs[0]]).t.c_min)
        max_v = max_v or "Expr({})".format(max_v or VI(self.value_info[self.inputs[0]]).t.c_max)
        return "clamp({{}}, {}, {})".format(min_v, max_v)
HalideGraphVisitor.register(HalideClipVisitor)

class HalideBinaryVisitor(HalideNodeVisitor):
    def generate_alg(self, dim_vars):
        ip0 = VI(self.value_info[self.inputs[0]])
        ip1 = VI(self.value_info[self.inputs[1]])

        ip0_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(ip0.shape,
                            dim_vars[-len(ip0.shape):])]
        ip0_expr     = self.generate_funcref("v_" + self.inputs[0], ip0_dim_vars)
        ip1_dim_vars = [(dvar if dim > 1 else "0") for dim, dvar in
                        zip(ip1.shape,
                            dim_vars[-len(ip1.shape):])]
        ip1_expr     = self.generate_funcref("v_" + self.inputs[1], ip1_dim_vars)

        expr         = self.expr.format(ip0_expr, ip1_expr)

        op0_expr     = self.generate_funcref("v_" + self.outputs[0], dim_vars)
        return [self.generate_assign(op0_expr, expr)]

class HalideAddVisitor(HalideBinaryVisitor):
    op_type = "Add"
    expr    = "{}+{}"
HalideGraphVisitor.register(HalideAddVisitor)

class HalideAndVisitor(HalideBinaryVisitor):
    op_type = "And"
    expr    = "({})&({})"
HalideGraphVisitor.register(HalideAndVisitor)

class HalideBitShiftVisitor(HalideBinaryVisitor):
    attr_fields = {"direction":("direction", "s", "")}

    op_type = "BitShift"

    @property
    def expr(self):
        return "{}<<{}" if self.direction_ == "LEFT" else "{}>>{}"
HalideGraphVisitor.register(HalideBitShiftVisitor)

class HalideArgMVisitor(HalideNodeVisitor):
    attr_fields = {"keepdims":("keepdims", "i", 1),
                   "axis"    :("axis"    , "i", 0)}

    @property
    def n_dim_vars(self):
        return len(VI(self.value_info[self.inputs[0]]).shape)

    def generate_alg(self, dim_vars):
        ip0 = VI(self.value_info[self.inputs[0]])
        op0 = VI(self.value_info[self.outputs[0]])

        if self.axis_ < 0:
            axis = len(ip0.shape) + self.axis_
        else:
            axis = self.axis_

        r_code, red_vars = self.generate_rdom("r",
                                      [[0, ip0.shape[axis]]])
        code = [r_code]

        op_dim_vars = list(dim_vars) if self.keepdims_ else \
                         [dvar for i, dvar in enumerate(dim_vars) \
                          if i != axis]

        ip_dim_vars = [(dvar if i != axis \
                        else red_vars[0]) \
                        for i, dvar in enumerate(dim_vars)]

        op_expr = self.generate_funcref("v_" + self.outputs[0], op_dim_vars)
        ip_expr = self.generate_funcref("v_" + self.inputs[0], ip_dim_vars)

        expr = "{}({})[0]".format(self.argm_type, ip_expr)
        expr = self.generate_cast(op0.t.c, expr)
        return code + [self.generate_assign(op_expr, expr)]


class HalideArgMaxVisitor(HalideArgMVisitor):
    op_type   = "ArgMax"
    argm_type = "argmax"
HalideGraphVisitor.register(HalideArgMaxVisitor)

class HalideArgMinVisitor(HalideArgMVisitor):
    op_type   = "ArgMin"
    argm_type = "argmin"
HalideGraphVisitor.register(HalideArgMinVisitor)


class HalidePoolVisitor(HalideNodeVisitor):
    attr_fields = {"kernel_shape"     :("kernel_shape"     , "ints", None),
                   "ceil_mode"        :("ceil_mode"        , "i"   , 0),
                   "count_include_pad":("count_include_pad", "i"   , 0),
                   "pads"             :("pads"             , "ints", None),
                   "auto_pad"         :("auto_pad"         , "s"   , ""),
                   "strides"          :("strides"          , "ints", None),
                   "storage_order"    :("storage_order"    , "i"   , 0)}

    @property
    def n_dim_vars(self):
        return len(VI(self.value_info[self.inputs[0]]).shape)

    def generate_alg(self, dim_vars):
        ip0 = VI(self.value_info[self.inputs[0]])

        pool_shape_   = list(self.kernel_shape_) if self.kernel_shape_ else ip0.shape[2:]
        strides_      = self.strides_ or [1 for ks in pool_shape_]
        if self.pads_:
            li        = len(self.pads_)//2
            pads_     = list(zip(self.pads_[:li],
                                 self.pads_[li:]))
        else:
            if self.auto_pad_ == "SAME_UPPER":
                pads_ = [(floor((ks-1)/2), ceil((ks-1)/2)) \
                         for ks in pool_shape_]
            elif self.auto_pad_ == "SAME_LOWER":
                pads_ = [(ceil((ks-1)/2), floor((ks-1)/2)) \
                         for ks in pool_shape_]
            else:
                pads_ = [(0, 0) for ks in pool_shape_]

        n_ign_dims_   = len(ip0.shape) - len(pool_shape_)

        code = []

        r_code, red_vars = self.generate_rdom("r",
                                              [(0, i) for i in pool_shape_])
        code.append(r_code)

        ip_vars = ["{}*{}+{}-{}".format(dv, st, rv, pad[0]) \
                   if rv else dv \
                   for (dv, rv, st, pad) in zip(
                           dim_vars,
                           [()]*n_ign_dims_ + red_vars,
                           [()]*n_ign_dims_ + strides_,
                           [()]*n_ign_dims_ + pads_)]

        p_code, padded = self.generate_padded(
            "pad", "v_" + self.inputs[0],
            self.pad_const,
            [(0, s) \
             if i >= n_ign_dims_ else \
             ("Expr()", "Expr()") \
             for i, s in enumerate(ip0.shape)])
        code.append(p_code)


        lhs = self.generate_funcref("v_" + self.outputs[0], dim_vars)
        r_code, rhs = self.generate_pool_rhs(dim_vars, red_vars, ip_vars, pool_shape_,
                                     self.generate_funcref(padded, ip_vars))
        code.extend(r_code)
        code.append(self.generate_assign(lhs, rhs))

        return code

class HalideAveragePoolVisitor(HalidePoolVisitor):
    op_type   = "AveragePool"
    pad_const = "0"
    def generate_pool_rhs(self, dim_vars, red_vars, ip_vars, pool_shape, padded_expr):
        ip0 = VI(self.value_info[self.inputs[0]])

        f_code, count_func = self.generate_funcdecl("count")
        code = [f_code]

        count_vars  = dim_vars[-len(red_vars):]
        count_expr  = self.generate_funcref(count_func,
                                            count_vars)
        if self.count_include_pad_:
            count_rhs = np.prod(pool_shape)
        else:
            f_code, ones_func = self.generate_funcdecl("ones")
            code.append(f_code)
            code.append(self.generate_assign(
                self.generate_funcref(ones_func,
                                      count_vars),
                "1"))
            f_code, padded_ones = self.generate_padded(
                "pad_ones", ones_func, "0",
                [(0, s) for s in \
                 ip0.shape[-len(red_vars):]])

            code.append(f_code)

            pad_ones_expr = self.generate_funcref(padded_ones,
                                                  ip_vars[-len(red_vars):])
            count_rhs = "sum({})".format(pad_ones_expr)
        code.append(self.generate_assign(count_expr, count_rhs))

        rhs_expr = "sum({}) / {}".format(
            padded_expr,
            self.generate_funcref(
                count_func,
                dim_vars[-len(red_vars):]))
        return code, rhs_expr
HalideGraphVisitor.register(HalideAveragePoolVisitor)


class HalideBaseConvVisitor(HalideNodeVisitor):
    attr_fields = {"kernel_shape": ("kernel_shape", "ints", None),
                   "pads"        : ("pads"        , "ints", None),
                   "strides"     : ("strides"     , "ints", None),
                   "dilations"   : ("dilations"   , "ints", None),
                   "group"       : ("group"       , "i"   , 1)}

    def generate_alg(self, dim_vars: List[str]) -> List[str]:
        w = VI(self.value_info[self.inputs[1]])
        ip0 = VI(self.value_info[self.inputs[0]])
        op0 = VI(self.value_info[self.outputs[0]])

        if self.pads_:
            li = len(self.pads_) // 2
            self.pads_      = list(zip(self.pads_[:li], self.pads_[li:]))
        kernel_shape_  = self.kernel_shape_ or w.shape[2:]
        dilations_     = self.dilations_ or [1] * len(kernel_shape_)
        pads_          = self.pads_ or [(0, 0)] * len(kernel_shape_)
        strides_       = self.strides_ or [1] * len(kernel_shape_)
        padded_        = sum(map(sum, pads_)) > 0


        code = []
        r_code, red_vars = self.generate_rdom(
            "r",
            [(0, i) for i in [w.shape[1]] + kernel_shape_])
        code.append(r_code)

        ip_vars  = [dim_vars[0], "{}+cast<int>(floor({}/{}))*{}".format(
            red_vars[0], dim_vars[1],
            op0.shape[1]//self.group_, ip0.shape[1]//self.group_)] + \
            ["{}*{}+{}*{}-{}".format(dv, stride, rv, dilation, pad[0]) for \
             dv, rv, pad, stride, dilation in \
             zip(dim_vars[2:], red_vars[1:], pads_, strides_, dilations_)]

        w_vars  = [dim_vars[1]] + red_vars

        if padded_:
            act_code, activation = self.generate_activation(dim_vars[:len(ip0.shape)])
            p_code, padded = self.generate_padded("padded",
                                                  activation,
                                                  0,
                                                  [("Expr()", "Expr()")] * 2 + \
                                                  [(0, s) for s in ip0.shape[2:]])
            code.extend(act_code)
            code.extend([p_code])
        else:
            p_code, padded = self.generate_funcdecl("padded")
            act_code, activation = self.generate_activation(dim_vars[:len(ip0.shape)])
            a_code = self.generate_assign(padded, activation)
            code.extend(act_code)
            code.extend([p_code, a_code])

        padded_expr = self.generate_funcref(padded, ip_vars)

        bias_expr = self.generate_bias(dim_vars[1])

        w_expr = self.generate_funcref("v_" + self.inputs[1], w_vars)

        rhs = self.generate_cast(op0.t.c,
                                 "sum({}*{})+{}".format(padded_expr, w_expr, bias_expr))
        code.append(self.generate_assign(
            self.generate_funcref("v_" + self.outputs[0], dim_vars),
            rhs))
        return code

class HalideConvVisitor(HalideBaseConvVisitor):
    op_type     = "Conv"

    def generate_bias(self, dim_var):
        if len(self.inputs) == 3:
            return self.generate_funcref("v_" + self.inputs[2], dim_vars)
        else:
            return "0"

    def generate_activation(self, dim_vars):
        return [], "v_" + self.inputs[0]

HalideGraphVisitor.register(HalideConvVisitor)

class HalideConvIntegerVisitor(HalideBaseConvVisitor):
    op_type     = "ConvInteger"

    def generate_bias(self, dim_var):
        return "0"

    def generate_activation(self, dim_vars):
        if len(self.inputs) >= 3 and self.inputs[2]:
            code, act = self.generate_funcdecl("a_{}".format(self.inputs[0]))
            code = [code,
                    self.generate_assign(self.generate_funcref(act, dim_vars),
                                         "{}-v_{}()".format(
                                             self.generate_funcref("v_" + self.inputs[0], dim_vars),
                                             self.inputs[2]))]

            return code, act
HalideGraphVisitor.register(HalideConvIntegerVisitor)

class HalideConvTransposeVisitor(HalideNodeVisitor):
    op_type     = "ConvTranspose"
    attr_fields = {"kernel_shape":("kernel_shape"  , "ints", None),
                   "pads"        :("pads"          , "ints", None),
                   "strides"     :("strides"       , "ints", None),
                   "dilations"   :("dilations"     , "ints", None),
                   "op_shape"    :("output_shape"  , "ints", None),
                   "op_pads"     :("output_padding", "ints", None),
                   "auto_pad"    :("auto_pad"      , "s"   , None)}
    def generate_alg(self, dim_vars):

        ip0 = VI(self.value_info[self.inputs[0]])
        ip1 = VI(self.value_info[self.inputs[1]])
        op  = VI(self.value_info[self.outputs[0]])
        idims = len(ip0.shape)
        wdims = len(ip1.shape)

        bias = self.inputs[2] if (len(self.inputs) == 3) else None


        strides   = self.strides_ or [1] * (wdims - 2)
        dilations = self.dilations_ or [1] * (wdims - 2)
        kernel_shape = self.kernel_shape_ or ip1.shape[2:]
        pads      = list(zip(self.pads_, self.pads_[::-1]))[:len(self.pads_)//2] if self.pads_ else [(0, 0)] * len(kernel_shape)
        op_pads   = self.op_pads_ or [0] * len(kernel_shape)
        if self.op_shape_ and self.auto_pad_:
            total_padding = [stride*(ops-1)+op_pad+ks-ips \
                             for (ips, ops, op_pad, ks, stride) in \
                             zip(ip0.shape[2:],
                                 op.shape[2:],
                                 op_pads,
                                 kernel_shape,
                                 strides)]
            if self.auto_pad_ != "SAME_UPPER":
                pads = [(tp//2, tp-tp//2) for tp in total_padding]
            else:
                pads = [(tp-tp//2, tp//2) for tp in total_padding]


        r_code, red_vars = self.generate_rdom("r", [(0, i) for i in ip0.shape[1:]])
        pw_code, pad_w = self.generate_padded("pad_w", "v_" + self.inputs[1], 0,
                                              [("Expr()","Expr()")]*2 + \
                                              [(0, s) for s in ip1.shape[2:]])
        pi_code, pad_i = self.generate_padded("pad_i", "v_" + self.inputs[0], 0,
                                              [("Expr()","Expr()")]*2 + \
                                              [(0, s) for s in ip0.shape[2:]])

        code = [r_code, pw_code, pi_code]

        d_code, dilated = self.generate_funcdecl("dilated")
        code.append(d_code)
        dilated_lhs = self.generate_funcref(dilated, dim_vars)
        dilated_cond = ["(({}%{})==0)".format(dv, dil) \
                        for dil, dv in zip(dilations, dim_vars[2:])]
        dilated_w_vars = ["cast<int>(floor({}/{}))".format(dv, dil) \
                          for dil, dv in zip(dilations, dim_vars[2:])]
        dilated_rhs = "select({}, {}, 0)".format(
            "&&".join(dilated_cond),
            self.generate_funcref(pad_w, dim_vars[:2] + dilated_w_vars))
        code.append(self.generate_assign(dilated_lhs, dilated_rhs))
        print(pads)
        ip_vars = [dim_vars[0], red_vars[0]] + \
                  ["cast<int>(floor(({0}-{4}*{5}+{2})/{1}))".format(dv, st, pad[0], op_pad, rv, dil) \
                   for dv, st, pad, op_pad, rv, dil \
                   in zip(dim_vars[2:],
                          strides,
                          pads,
                          op_pads,
                          red_vars[1:],
                          dilations)]
        w_vars = [red_vars[0], dim_vars[1]] + \
                 ["{}*{}".format(rv, dil) for \
                  dv, rv, pad, dil, stride in \
                  zip(dim_vars[2:],
                      red_vars[1:],
                      pads,
                      dilations,
                      strides)]
        sel_expr = ["((({0}-{2}*{3}+{5})%{1})==0)".format(dv, st, rv, dil, op_pad, pad[0]) \
                    for dv, rv, st, dil, op_pad, pad in \
                    zip(dim_vars[2:],
                        red_vars[1:],
                        strides,
                        dilations,
                        op_pads,
                        pads)]
        if bias:
            bias_expr = "+v_{}".format(self.generate_funcref(bias,
                                                             [dim_vars[1]]))
        else:
            bias_expr = ""

        lhs = self.generate_funcref("v_" + self.outputs[0], dim_vars)
        rhs = "sum(select({},{},0)*{}){}".format(
            "&&".join(sel_expr),
            self.generate_funcref(pad_i, ip_vars),
            self.generate_funcref(dilated, w_vars),
            bias_expr)
        code.append(self.generate_assign(lhs, rhs))
        return code
HalideGraphVisitor.register(HalideConvTransposeVisitor)

class HalideBatchNormVisitor(HalideNodeVisitor):
    op_type = "BatchNormalization"
    attr_fields = {"eps"   : ("epsilon", "f"   , 1e-5),
                   "eps_t" : ("epsilon", "type", None)}

    def generate_alg(self, dim_vars: List[str]) -> List[str]:
        x    = self.inputs[0]
        s    = self.inputs[1]
        bias = self.inputs[2]
        mean = self.inputs[3]
        var  = self.inputs[4]
        op0  = VI(self.value_info[self.outputs[0]])

        lhs    = self.generate_funcref("v_" + self.outputs[0], dim_vars)

        s_expr    = self.generate_funcref("v_" + s, [dim_vars[1]])
        x_expr    = self.generate_funcref("v_" + x, dim_vars)
        mean_expr = self.generate_funcref("v_" + mean, [dim_vars[1]])
        var_expr  = self.generate_funcref("v_" + var, [dim_vars[1]])
        bias_expr = self.generate_funcref("v_" + bias, [dim_vars[1]])
        eps_expr  = self.generate_cast(op0.t.c, self.eps_)
        rhs = "{}*(({}-{})/sqrt({}+{}))+{}".format(
            s_expr, x_expr, mean_expr, var_expr, eps_expr, bias_expr)
        return [self.generate_assign(self.generate_funcref(
            "v_" + self.outputs[0],
            dim_vars), rhs)]
HalideGraphVisitor.register(HalideBatchNormVisitor)

class HalideConcatVisitor(HalideNodeVisitor):
    op_type = "Concat"
    attr_fields = {"axis" : ("axis", "i", 0)}


    def generate_alg(self, dim_vars):
        code = [self.generate_assign(
            self.generate_funcref("v_" + self.outputs[0], dim_vars),
            "undef<{}>()".format(VI(self.value_info[self.outputs[0]]).t.c))]
        prev_s = 0
        for i, ip in enumerate(self.inputs):
            ipvi = VI(self.value_info[ip])

            r_code, red_var = self.generate_rdom(str(i),
                                                 [(prev_s, ipvi.shape[self.axis_])])
            red_var = red_var[0]
            code.append(r_code)
            op_vars = list(dim_vars)
            op_vars[self.axis_] = red_var
            ip_vars = list(dim_vars)
            ip_vars[self.axis_] = "{}-{}".format(red_var, prev_s)
            prev_s += ipvi.shape[self.axis_]
            code.append(self.generate_assign(
                self.generate_funcref("v_" + self.outputs[0], op_vars),
                self.generate_funcref("v_" + ip, ip_vars)))
        return code
HalideGraphVisitor.register(HalideConcatVisitor)

class HalideDepthToSpaceVisitor(HalideNodeVisitor):
    op_type = "DepthToSpace"
    attr_fields = {"blocksize":("blocksize", "i", None)}
    def generate_alg(self, dim_vars):
        ip0 = VI(self.value_info[self.inputs[0]])
        ip_vars = [dim_vars[0],
                   "{}+({}%{})*{}+({}%{})*{}".format(
                       dim_vars[1],
                       dim_vars[3], self.blocksize_,
                       ip0.shape[1] // (self.blocksize_**2),
                       dim_vars[2], self.blocksize_,
                       ip0.shape[1] // self.blocksize_),
                   "cast<int>({}/{})".format(
                       dim_vars[2],
                       self.blocksize_),
                   "cast<int>({}/{})".format(
                       dim_vars[3],
                       self.blocksize_)]
        return [self.generate_assign(self.generate_funcref("v_" + self.outputs[0],
                                                           dim_vars),
                                     self.generate_funcref("v_" + self.inputs[0],
                                                           ip_vars))]
HalideGraphVisitor.register(HalideDepthToSpaceVisitor)

class HalidePadVisitor(HalideNodeVisitor):
    op_type = "Pad"
    attr_fields = {"mode"  :("mode","s","constant")}

    def generate_alg(self, dim_vars):
        ip = VI(self.value_info[self.inputs[0]])
        op = VI(self.value_info[self.outputs[0]])

        lhs = "Func {}_pad".format(self.outputs[0])
        if self.mode_ == "constant":
            if len(self.inputs) >= 2 and self.inputs[2]:
                const = "v_{}()".format(self.inputs[2])
            else:
                const = self.generate_cast(op.t.c, "Expr(0)")
            rhs = "constant_exterior(v_{},{},{{{}}})".format(
                self.inputs[0], const,
                JOIN_VARS(["{{0,{}}}".format(ips) for ips in ip.shape]))
        elif self.mode_ == "edge":
            rhs = "repeat_edge(v_{},{{{}}})".format(
                self.inputs[0],
                JOIN_VARS(["{{0,{}}}".format(ips) for ips in ip.shape]))
        elif self.mode_ == "reflect":
            rhs = "mirror_interior(v_{},{{{}}})".format(
                self.inputs[0],
                JOIN_VARS(["{{0,{}}}".format(ips) if ips > 1 else "{Expr(),Expr()}"\
                          for ips in ip.shape]))
        code = [self.generate_assign(lhs, rhs)]

        pads = [self.generate_cast("int", "v_{}({})".format(self.inputs[1], i)) for i in range(2*len(ip.shape))]
        pads = list(zip(pads, pads[::-1]))[:len(pads)//2]
        n_ign_dims = len(ip.shape) - len(pads)

        ip_vars = ["{}-{}".format(dv, pad[0]) if pad else dv \
                   for dv, pad in zip(dim_vars, [{}]*n_ign_dims+pads)]

        op_expr = self.generate_funcref("v_" + self.outputs[0], dim_vars)
        ip_expr = self.generate_funcref("{}_pad".format(self.outputs[0]),
                                        ip_vars)
        code.append(self.generate_assign(op_expr, ip_expr))

        return code
HalideGraphVisitor.register(HalidePadVisitor)


class HalideSqueezeVisitor(HalideNodeVisitor):
    op_type = "Squeeze"
    attr_fields = {"axes":("axes", "ints", None)}

    def generate_alg(self, dim_vars: List[str]):
        op0 = VI(self.value_info[self.outputs[0]])
        op0_expr = self.generate_funcref("v_" + self.outputs[0], dim_vars)

        ip0 = VI(self.value_info[self.inputs[0]])
        ip_vars = ["0"] * len(ip0.shape)
        for i, dv in zip([i for i in range(len(ip0.shape)) \
                          if i not in self.axes_],
                         dim_vars):
            ip_vars[i] = dv
        ip0_expr = self.generate_funcref("v_" + self.inputs[0], ip_vars)

        assgn = self.generate_assign(op0_expr, ip0_expr)
        return [assgn]
HalideGraphVisitor.register(HalideSqueezeVisitor)

class HalideTransposeVisitor(HalideNodeVisitor):
    op_type = "Transpose"
    attr_fields = {"perms":("perm","ints",None)}

    def generate_alg(self, dim_vars: List[str]):
        ip0 = VI(self.value_info[self.inputs[0]])
        ip0_expr = self.generate_funcref("v_" + self.inputs[0], dim_vars)

        op0 = VI(self.value_info[self.outputs[0]])
        perms = self.perms_ or range(ip0.dims)[::-1]
        op0_dim_vars = [ip0.shape[i] for i in perms]
        op0_expr = self.generate_funcref("v_" + self.outputs[0],  [dim_vars[i] for i in perms])

        assgn = self.generate_assign(op0_expr, ip0_expr)
        return [assgn]
HalideGraphVisitor.register(HalideTransposeVisitor)

class HalideUnsqueezeGenerator(HalideNodeVisitor):
    op_type = "Unsqueeze"
    attr_fields = {"axes":("axes","ints",None)}

    def generate_alg(self, dim_vars: List[str]):
        op0 = VI(self.value_info[self.outputs[0]])
        op0_expr = self.generate_funcref("v_" + self.outputs[0], dim_vars)

        ip0 = VI(self.value_info[self.inputs[0]])
        op_shape = ip0.shape
        orig_s_ = [1] * ip0.dims
        for i in self.axes_:
            op_shape.insert(i, 1)
            orig_s_.insert(i, 0)
        ip_vars = [dv for i, dv in enumerate(dim_vars) if orig_s_[i]]
        ip0_expr = self.generate_funcref("v_" + self.inputs[0], ip_vars)
        assgn = self.generate_assign(op0_expr, ip0_expr)

        return [assgn]
HalideGraphVisitor.register(HalideUnsqueezeGenerator)

class HalideReshapeGenerator(HalideNodeVisitor):
    op_type = "Reshape"

    def generate_alg(self, dim_vars: List[str]):
        op0 = VI(self.value_info[self.outputs[0]])
        op0_expr = self.generate_funcref("v_" + self.outputs[0], dim_vars)

        ip0 = VI(self.value_info[self.inputs[0]])
        prevs = ip0.shape[1:] + [1]
        prods = [np.prod(prevs[i:]) for i in range(len(prevs))]
        ip_vars = ["cast<int>(floor({}/{}))%{}".format(dim_vars[0],
                                                       prod,
                                                       ips) \
                   for ips, prod in \
                   zip(ip0.shape, prods)]
        ip0_expr = self.generate_funcref("v_" + self.inputs[0], ip_vars)
        flattened_decl, flattened_name = self.generate_funcdecl("flattened")
        code = [flattened_decl]

        flattened_expr = self.generate_funcref(flattened_name, [dim_vars[0]])
        assgn1 = self.generate_assign(flattened_expr, ip0_expr)

        prevs = op0.shape[1:] + [1]
        prods = [np.prod(prevs[i:]) for i in range(len(prevs))]
        fl_vars = ["({}*{})".format(dv, p) for dv, p in zip(dim_vars, prods)]

        assgn2 = self.generate_assign(op0_expr, self.generate_funcref(flattened_name, ["+".join(fl_vars)]))
        return [flattened_decl, assgn1, assgn2]
HalideGraphVisitor.register(HalideReshapeGenerator)
