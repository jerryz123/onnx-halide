from .base_generator import BaseGraphVisitor, BaseNodeVisitor
from .types import MasterType
import os
from os.path import join, dirname
import subprocess
from . import __path__

class HalideGraphVisitor(BaseGraphVisitor):
    pass

def JOIN_VARS(strs):
    return ','.join(strs[::-1])


class HalideNodeVisitor(BaseNodeVisitor):
    @property
    def n_dim_vars(self):
        pass

    def __init__(self):
        halide_runtime = join(dirname(__path__[0]), "runtime/HalideRuntime.o")
        BaseGraphVisitor.register_runtime({halide_runtime}, set())


    def generate_alg(self, dim_vars):
        pass

    def visit(self, node, value_info):
        BaseNodeVisitor.visit(self, node, value_info)
        dim_vars = ["d{}".format(i) for i in range(self.n_dim_vars)]

        gen_name = "HalideNode_{}".format(self.outputs[0])

        input_decls = '\n'.join(["Input<Buffer<{0}>> {1}{{\"{1}\", {2}}};".format(
            MasterType.from_onnx(value_info[i].tensor_type.elem_type).c_t,
            i,
            len(value_info[i].tensor_type.shape.dim),
            list(value_info[i].tensor_type.shape.dim)) for i in self.inputs])

        output_decls = '\n'.join(["Output<Buffer<{0}>> {1}{{\"{1}\", {2}}};".format(
            MasterType.from_onnx(value_info[i].tensor_type.elem_type).c_t,
            i,
            len(value_info[i].tensor_type.shape.dim),
            list(value_info[i].tensor_type.shape.dim)) for i in self.outputs])

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
        src = src.format(gen_name, input_decls, output_decls, dim_var_decls, alg)

        src_fname = join(self.temp_dir, "{}.cpp".format(gen_name))
        generator_bin = join(self.temp_dir, "{}.bin".format(gen_name))
        with open(src_fname, 'w') as f:
            f.write(src)


        cmd  = "{} -std=c++11 ".format(self.cxx)
        cmd += "-I {} -I {} ".format(join(self.install_dir, "include"),
                                     join(self.install_dir, "share/halide/tools"))
        cmd += "-fno-rtti "
        cmd += "{} {} {} ".format(src_fname,
                                  join(self.install_dir, "lib/libHalide.so"),
                                  join(self.install_dir, "share/halide/tools/GenGen.cpp"))
        cmd += "-o {} -ldl -lpthread -lz ".format(generator_bin)
        cmd += "-lrt -ldl -ltinfo -lm"

        r = subprocess.run(cmd, check=True, shell=True)
        print(cmd)

        cmd  = "./{} -g {} -o {} ".format(generator_bin, gen_name, self.temp_dir)
        cmd += "-e h,o "
        cmd += "target=riscv-64-noos-no_asserts-no_runtime"

        r = subprocess.run(cmd, check=True, shell=True)

        srcc = """
#include "{0}.h"
#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include <cfenv>
using float16_t = uint16_t;
using namespace Halide::Runtime;
extern "C" {{

int {0}_c_func({1}) {{

{2}

int r = {0}({3});

return r;
}};

}}
"""

        
        cargs    = []
        buffers  = []
        haargs   = []

        for i in self.inputs + self.outputs:
            ttype = value_info[i].tensor_type
            ctype = MasterType.from_onnx(ttype.elem_type).c_t
            shape = [d.dim_value for d in ttype.shape.dim]

            cargs.append("{}* {}".format(
                ctype,
                i))
            buffers.append("Buffer<{0}> {1}_buf({1}, {{{2}}});".format(
                ctype,
                i,
                JOIN_VARS([str(i) for i in shape])))
            haargs.append("{}_buf".format(i))

        srcc = srcc.format(gen_name,
                           ','.join(cargs),
                           '\n'.join(buffers),
                           ','.join(haargs))

        srcc_fname = join(self.temp_dir, "{}.c".format(gen_name))
        with open(srcc_fname, 'w') as f:
            f.write(srcc)

        cmd  = "{} -std=c++11 ".format(self.rvcxx)
        cmd += "-I {} -I {} -I {} -fno-rtti ".format(join(self.install_dir, "include"),
                                                     join(self.install_dir, "share/halide/tools"),
                                                     self.temp_dir)
        cmd += "-march=rv64imafdc -mabi=lp64d "
        cmd += "-c {} -o {} ".format(srcc_fname,
                                     join(self.temp_dir, "{}_c.o".format(gen_name)))

        r = subprocess.run(cmd, check=True, shell=True)

        objects = {join(self.temp_dir, "{}.o".format(gen_name)),
                   join(self.temp_dir, "{}_c.o".format(gen_name))}

        headers = {join(self.temp_dir, "{}.h".format(gen_name))}

        code = "{}_c_func({});".format(
            gen_name,
            ','.join(cargs))
        return [code], objects, headers


    def generate_funcref(self, func, dim_vars):
        return "{}({})".format(func, JOIN_VARS(dim_vars))

    def generate_assign(self, lhs, dim_vars, rhs):
        return "{} = {};".format(self.generate_funcref(lhs, dim_vars),
                                 rhs)

class HalideUnaryVisitor(HalideNodeVisitor):
    @property
    def n_dim_vars(self):
        return len(self.value_info[self.outputs[0]].tensor_type.shape.dim)

    def generate_alg(self, dim_vars):
        ip0_expr  = self.generate_funcref(self.inputs[0], dim_vars)
        unop_expr = self.expr.format(ip0_expr)
        return self.generate_assign(self.outputs[0], dim_vars, unop_expr)


class HalideAbsVisitor(HalideUnaryVisitor):
    op_type = "Abs"
    expr    = "abs({})"

HalideGraphVisitor.register(HalideAbsVisitor)

