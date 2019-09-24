from .base_generator import BaseGenerator, BaseNode, REGISTERNODE
from .types import MasterType

from os.path import join
import subprocess

class HalideGenerator(BaseGenerator):
    node_lookup = {}
    def __init__(self, temp_dir, cxx, install_dir):
        self.install_dir = install_dir
        self.cxx = cxx
        self.temp_dir = temp_dir

    def generate(self, node, value_info):
        return self.node_lookup[node.op_type](node, value_info).compile(self.temp_dir, self.cxx, self.install_dir)

class HalideFunc:
    def __init__(self, name, shape, type, io):
        self.name = name
        self.shape = tuple([int(i) for i in shape])
        self.type = type
        self.io = io

        self.dims = len(shape)

class HalideNode(BaseNode):
    @property
    def n_dim_vars(self):
        pass

    def compile(self, temp_dir, cxx, install_dir):
        dim_vars = ["d{}".format(i) for i in range(self.n_dim_vars)]

        gen_name = "HalideNode_{}".format(self.outputs[0])

        input_decls = '\n'.join(["Input<Buffer<{0}>> {1}{{\"{1}\", {2}}};".format(
            MasterType.from_onnx(self.value_info[i].tensor_type.elem_type).c_t,
            i,
            len(self.value_info[i].tensor_type.shape.dim),
            list(self.value_info[i].tensor_type.shape.dim)) for i in self.inputs])

        output_decls = '\n'.join(["Output<Buffer<{0}>> {1}{{\"{1}\", {2}}};".format(
            MasterType.from_onnx(self.value_info[i].tensor_type.elem_type).c_t,
            i,
            len(self.value_info[i].tensor_type.shape.dim),
            list(self.value_info[i].tensor_type.shape.dim)) for i in self.outputs])

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

        src_fname = join(temp_dir, "{}.cpp".format(gen_name))
        generator_bin = join(temp_dir, "{}.bin".format(gen_name))
        with open(src_fname, 'w') as f:
            f.write(src)


        cmd  = "{} -std=c++11 ".format(cxx)
        cmd += "-I {} -I {} ".format(join(install_dir, "include"), join(install_dir, "share/halide/tools"))
        cmd += "-fno-rtti "
        cmd += "{} {} {} ".format(src_fname,
                                  join(install_dir, "lib/libHalide.so"),
                                  join(install_dir, "share/halide/tools/GenGen.cpp"))
        cmd += "-o {} -ldl -lpthread -lz ".format(generator_bin)
        cmd += "-lrt -ldl -ltinfo -lm"

        r = subprocess.run(cmd, check=True, shell=True)

        cmd  = "./{} -g {} -o {} ".format(generator_bin, gen_name, temp_dir)
        cmd += "-e h,o "
        cmd += "target=riscv-64-noos-no_asserts-no_runtime"

        r = subprocess.run(cmd, check=True, shell=True)

        cmd  = "./{} -r {} -o {} ".format(generator_bin, "halide_runtime", temp_dir)
        cmd += "-e h,o "
        cmd += "target=riscv-64-noos-no_asserts"

        r = subprocess.run(cmd, check=True, shell=True)

        objects = {join(temp_dir, "{}.o".format(gen_name)),
                   join(temp_dir, "halide_runtime.o")}
        headers = {join(temp_dir, "{}.h".format(gen_name)),
                   join(temp_dir, "halide_runtime.h")}


        return None, objects, headers


    def generate_funcref(self, func, dim_vars):
        return "{}({})".format(func, ','.join(dim_vars[::-1]))

    def generate_assign(self, lhs, dim_vars, rhs):
        return "{} = {};".format(self.generate_funcref(lhs, dim_vars),
                                 rhs)

class UnaryNode(HalideNode):
    @property
    def n_dim_vars(self):
        return len(self.value_info[self.outputs[0]].tensor_type.shape.dim)

    def generate_alg(self, dim_vars):
        ip0_expr  = self.generate_funcref(self.inputs[0], dim_vars)
        unop_expr = self.expr.format(ip0_expr)
        return self.generate_assign(self.outputs[0], dim_vars, unop_expr)


class AbsNode(UnaryNode):
    op_type = "Abs"
    expr    = "abs({})"

REGISTERNODE(AbsNode, HalideGenerator)
