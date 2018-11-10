import numpy as np
from onnx.backend.base import Backend
from onnx import helper, TensorProto

class HalideBackend(Backend):
    # @staticmethod
    # def set_device(device):
    #     if device == 'CPU':
    #         C.try_set_default_device(C.device.cpu())
    #     elif device == 'GPU' or device == 'CUDA':
    #         try:
    #             C.try_set_default_device(C.device.gpu(0))
    #         except:
    #             C.use_default_device()
    #     else:
    #         C.use_default_device()

    @classmethod
    def run_node(cls, node, input, device='CPU'):
        input_tensors = []

        if len(node.input) != len(input):
            raise ValueError(
                "Unexpected Input Size: Op_Type = {0}, Expected = {1}, Received = {2}"
                .format(node.op_type, len(node.input), len(input))
                ) 

        for i in range(len(input)):
            input_tensors.append(helper.make_tensor_value_info(node.input[i], TensorProto.FLOAT, input[i].shape))

        onnx_graph = helper.make_graph([node], "test_{}".format(node.op_type), input_tensors, [])
        onnx_model = helper.make_model(onnx_graph)
        return HalideBackend.run_model(onnx_model, input, device)

    @classmethod
    def generate_halide(cls, model):

        halide_str = """"""
        buffers = {}
        print(type(model.graph))
        for init in model.graph.initializer:
            dims      = init.dims
            data_type = init.data_type
            name      = init.name
            raw_data  = init.raw_data

            s_data_type = {k: v for (v, k) in TensorProto.DataType.items()}[init.data_type]
            c_typ = {"FLOAT" : "float*"}[s_data_type]
            if init.raw_data:
                halide_str += "char* raw_{} = {};\n".format(init.name, init.raw_data)
                halide_str += "{0} {1} = ({0}) raw_{1};\n".format(c_typ, init.name, init.name)
                #halide_str += "{} {};".format(c_typ, init.name)

        return halide_str
    @classmethod
    def run_model(cls, model, input, device='CPU'):
        halide_str = HalideBackend.generate_halide(model)
        print(halide_str)
        raise NotImplementedError

    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        raise NotImplementedError
    @classmethod
    def supports_device(cls, device='CPU'):
        return device in ['CPU']


run_model = HalideBackend.run_model
run_node = HalideBackend.run_node
supports_device = HalideBackend.supports_device
prepare = HalideBackend.prepare
