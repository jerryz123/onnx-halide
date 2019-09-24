class BaseGenerator:
    node_lookup = {}

    def generate(self, node, value_info):
        pass

class BaseNode:
    op_type = ""
    def __init__(self, node, value_info):
        self.node = node
        self.value_info = value_info

        self.inputs = list(node.input)
        self.outputs = list(node.output)

    def compile(self, node, value_info):
        pass

def REGISTERNODE(node, generator):
    generator.node_lookup[node.op_type] = node
