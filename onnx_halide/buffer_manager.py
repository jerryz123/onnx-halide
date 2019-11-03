from .types import VI
from typing import Dict
from onnx.onnx_ml_pb2 import TypeProto


class BufferManager:
    def __init__(self, value_info: Dict[str, TypeProto]):
        self.value_info = value_info
    '''Tracks and allocates buffers, reusing them when possible.
    Currently supported modes are "naive" (always allocate new),
    and "double buffer'''
    def allocate_buffer(self, op: str) -> str:
        raise Exception('Cannot use abstract allocator ')

class Buffer:
    bufferCount = 0
    @classmethod
    def allocBuffer(cls) -> str:
        cls.bufferCount += 1
        return "buffer{}".format(str(cls.bufferCount))
    def __init__(self, c_type: str, size: int, name = None):
        self.c_type = c_type
        self.size = size
        self.name = name if name is not None else Buffer.allocBuffer()

class NaiveBufferManager(BufferManager):
    '''Allocate a new buffer for each output'''

    def allocate_buffer(self, op: str) -> str:
        op_shape = VI(self.value_info[op]).shape
        return "  {0}* {1} = ({0}*) malloc({2}*sizeof({0}));".format(
            VI(self.value_info[op]).t.c,
            op,
            "*".join(map(str, op_shape)) if op_shape else "1")