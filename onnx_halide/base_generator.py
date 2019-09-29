from .types import VI
import os
from os.path import join, abspath
import subprocess

from onnx.onnx_ml_pb2 import GraphProto, NodeProto, TypeProto
from typing import Any, Dict, List, Set, Tuple
class BaseVisitor:
    install_dir = os.environ['RISCV']
    cxx = "g++"
    def __init__(self, temp_dir: str = "temp") -> None:
        self.temp_dir = abspath(temp_dir)
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    # Returns c call, object files, headers
    def visit(self, graph_or_node, value_info):
        pass


# Does a linear traversal of the model
# according to nodes in the node_lookup map
# Inheritors of this will likely want to override
# visit() to implement their own scheduling
class BaseGraphVisitor(BaseVisitor):
    node_lookup = {}
    runtime_objects = set()
    runtime_headers = set()

    def __init__(self, **kwargs) -> None:
        BaseVisitor.__init__(self, **kwargs)
        self.objects = set()
        self.headers = set()

    @classmethod
    def register(cls, node_class: Any) -> None:
        cls.node_lookup[node_class.op_type] = node_class

    @classmethod
    def register_runtime(cls, objects: Set[str], headers: Set[Any]) -> None:
        cls.runtime_objects |= objects
        cls.runtime_headers |= headers

    def visit(self, graph: GraphProto, value_info: Dict[str, TypeProto]) -> Tuple[List[str], List[str], Set[str]]:
        inputs = [i.name for i in list(graph.input)]
        outputs = [i.name for i in list(graph.output)]

        name = graph.name

        code = []

        for node in graph.node:
            node_outputs = list(node.output)
            generator = self.node_lookup[node.op_type](temp_dir=self.temp_dir)
            node_code, objects, headers = generator.visit(node, value_info)
            self.objects |= objects
            self.headers |= headers

            for op in list(node.output):
                if op not in outputs:
                    code.append("  {} v_{}[{}];".format(
                        VI(value_info[op]).t.c,
                        op,
                        "*".join(VI(value_info[op]).shape)))

            for c in node_code:
                code.append("  " + c)

        cargs = []
        for vi in list(graph.input) + list(graph.output):
            name = vi.name
            vi   = VI(vi.type)
            cargs.append("{}* v_{}".format(vi.t.c, name))

        code = ["void {}({}) {{".format(graph.name, ','.join(cargs))] + \
               code + \
               ["};"]

        api_header = '\n'.join(["#ifndef {}_H".format(graph.name),
                                "#define {}_H".format(graph.name),
                                "#include <stdint.h>".format(graph.name),
                                "void {}({});".format(graph.name, ','.join(cargs)),
                                "#endif"])
        api_header_fname = join(self.temp_dir, "{}.h".format(graph.name))
        with open(api_header_fname, 'w') as f:
            f.write(api_header)

        return (code,
                list(self.runtime_objects) + list(self.objects),
                self.headers | self.runtime_headers | {"\"{}\"".format(api_header_fname)})


class BaseNodeVisitor(BaseVisitor):
    op_type = ""
    def visit(self, node: NodeProto, value_info: Dict[str, TypeProto]) -> None:
        assert(node.op_type == self.op_type)
        self.node = node
        self.value_info = value_info

        self.inputs  = list(node.input)
        self.outputs = list(node.output)

        pass
