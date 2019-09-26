from .types import MasterType
import os
from os.path import join
import subprocess

class BaseVisitor:
    temp_dir = "temp"
    install_dir = os.environ['RISCV']
    cxx = "g++"
    rvcxx = "riscv64-unknown-elf-c++"

    # Returns c call, object files, headers
    def visit(self, graph_or_node, value_info):
        pass


# Does a linear traversal of the model
# according to nodes in the node_lookup map
# Inheritors of this will likely want to override
# generate() to implement their own scheduling
class BaseGraphVisitor(BaseVisitor):
    node_lookup = {}
    runtime_objects = set()
    runtime_headers = set()

    def __init__(self):

        self.objects = set()
        self.headers = set()

    @classmethod
    def register(cls, node_class):
        cls.node_lookup[node_class.op_type] = node_class

    @classmethod
    def register_runtime(cls, objects, headers):
        cls.runtime_objects |= objects
        cls.runtime_headers |= headers

    def visit(self, graph, value_info):
        inputs = [i.name for i in list(graph.input)]
        outputs = [i.name for i in list(graph.output)]

        name = graph.name

        body = []
        src = """
void {0}({1}) {{

{2}

}};

"""

        for node in graph.node:
            node_outputs = list(node.output)
            generator = self.node_lookup[node.op_type]()
            code, objects, headers = generator.visit(node, value_info)
            self.objects |= objects
            self.headers |= headers

            for op in list(node.output):
                if op not in outputs:
                    body.append("{} {}[{}];".format(MasterType.from_onnx(value_info[op].tensor_type.elem_type).c_t,
                                                    op,
                                                    "*".join([str(d.dim_value) for d in value_info[op].tensor_type.shape.dim])))
            body.extend(code)

        cargs = []
        for i in inputs + outputs:
            ttype = value_info[i].tensor_type
            ctype = MasterType.from_onnx(ttype.elem_type).c_t
            cargs.append("{}* {}".format(ctype, i))
        src = src.format(graph.name,
                         ','.join(cargs),
                         '\n'.join(body))


        src_fname = join(self.temp_dir, "{}.c".format(name))

        with open(src_fname, 'w') as f:
            f.write(src)

        cmd  = "{} -std=c++11 ".format(self.rvcxx)
        cmd += "-fno-rtti "
        cmd += "-march=rv64imafdc -mabi=lp64d "
        cmd += "-c {} -o {}".format(src_fname,
                                    join(self.temp_dir, "{}.o".format(graph.name)))
        #r = subprocess.run(cmd, check=True, shell=True)

        code = "{}({});".format(graph.name, ','.join(cargs));
        return [code], list(self.runtime_objects) + list(self.objects), self.headers | self.runtime_headers


class BaseNodeVisitor(BaseVisitor):
    op_type = ""
    def visit(self, node, value_info):
        assert(node.op_type == self.op_type)
        self.node = node
        self.value_info = value_info

        self.inputs  = list(node.input)
        self.outputs = list(node.output)
        pass
