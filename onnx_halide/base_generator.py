import os
import subprocess
import numpy as np

from os.path import join, abspath
from onnx.onnx_ml_pb2 import GraphProto, NodeProto, TypeProto
from typing import Any, Dict, List, Set, Tuple

from .types import VI
from .environment_link import Environment
from .buffer_manager import NaiveBufferManager

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
    runtime_headers = set(["<stdlib.h>", "<stdio.h>"])

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
        buffer_manager = NaiveBufferManager(value_info)
        for node in graph.node:
            node_outputs = list(node.output)
            generator = self.node_lookup[node.op_type](temp_dir=self.temp_dir)
            node_code, objects, headers = generator.visit(node, value_info)
            self.objects |= objects
            self.headers |= headers

            for op in list(node.output):
                if op not in outputs and op in value_info:
                    code.append(buffer_manager.allocate_buffer(op))

            for c in node_code:
                code.append("  " + c)

        cargs = []
        for vi in list(graph.input) + list(graph.output):
            name = vi.name
            vi   = VI(vi.type)
            cargs.append("{}* {}".format(vi.t.c, name))

        code = ["void {}({}) {{".format(graph.name, ','.join(cargs))] + \
               code + \
               ["};"]

        api_header = '\n'.join(["#ifndef {}_H".format(graph.name),
                                "#define {}_H".format(graph.name),
                                "#include <stdint.h>".format(graph.name),
                                "#define float16_t uint16_t",
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



class ConstantVisitor(BaseNodeVisitor):
    op_type = "Constant"
    attr_fields = {"tensor":("value","t",None)}

    def visit(self, node, value_info):
        BaseNodeVisitor.visit(self, node, value_info)
        op = VI(value_info[self.outputs[0]])
        data = None
        if self.tensor_.raw_data:
            data = np.frombuffer(self.tensor_.raw_data,
                                 count=op.size,
                                 dtype=op.t.np)
        else:
            data = np.array(list(self.tensor_.float_data)) \
                     .astype(op.t.np)

        gen_name = "constant_{}".format(self.outputs[0])
        ofile, hfile, ref_name = Environment.compile_constant_object(
            gen_name, data, self.temp_dir)


        # TODO: Don't do memcpy. In that case just manipulate the pointer
        code = ["memcpy({0}, {1}, {1}_len);".format(self.outputs[0], ref_name)]


        return code, {ofile}, {"\"{}\"".format(hfile), "<string.h>"}
BaseGraphVisitor.register(ConstantVisitor)

class ConstantOfShapeVisitor(BaseNodeVisitor):
    op_type = "ConstantOfShape"
    attr_fields = {"value":("value", "t", None)}

    def visit(self, node, value_info):
        BaseNodeVisitor.visit(self, node, value_info)
        op = VI(value_info[self.outputs[0]])
        if not self.value_:
            value = np.array([0.0]).astype(float)
        else:
            data = None
            # TODO fill out this switch
            if op.t.c == "float":
                data = self.value_.float_data
            elif op.t.c == "int32_t":
                data = self.value_.int32_data
            value = np.array(data).astype(op.t.np)
        data = np.full(op.shape, value)

        gen_name = "constant_{}".format(self.outputs[0])

        ofile, hfile, ref_name = Environment.compile_constant_object(
            gen_name, data, self.temp_dir)

        code = ["memcpy({0}, {1}, {1}_len);".format(
            self.outputs[0], ref_name)]
        return code, {ofile}, {"\"{}\"".format(hfile), "<string.h>"}
BaseGraphVisitor.register(ConstantOfShapeVisitor)

class EyeLikeVisitor(BaseNodeVisitor):
    op_type = "EyeLike"
    attr_fields = {"k":("k","i",0)}

    def visit(self, node, value_info):
        BaseNodeVisitor.visit(self, node, value_info)
        op = VI(value_info[self.outputs[0]])

        data = np.eye(op.shape[0],
                      op.shape[1],
                      k=self.k_,
                      dtype=op.t.np)
        gen_name = "eye_{}".format(self.outputs[0])
        ofile, hfile, ref_name = Environment.compile_constant_object(
            gen_name, data, self.temp_dir)

        code = ["memcpy({0}, {1}, {1}_len);".format(
            self.outputs[0], ref_name)]
        return code, {ofile}, {"\"{}\"".format(hfile), "<string.h>"}
BaseGraphVisitor.register(EyeLikeVisitor)
