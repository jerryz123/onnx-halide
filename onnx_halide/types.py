import numpy as np
from onnx import TensorProto


from onnx.onnx_ml_pb2 import TypeProto
from typing import Any, List
class MasterType:
    onnx_int_dict = {}
    c_dict = {}
    def __init__(self, onnx_str: str, c: str, numpy: Any, min: str, max: str) -> None:
        self.onnx_int    = {k:v for k,v in TensorProto.DataType.items()}[onnx_str]
        self.onnx_str    = onnx_str
        self.c           = c
        self.np          = numpy
        self.c_min       = min
        self.c_max       = max
        MasterType.onnx_int_dict[self.onnx_int] = self
        MasterType.c_dict[self.c] = self

    def __eq__(self, other):
        return self.onnx_int == other.onnx_int

    @classmethod
    def from_c(c):
        return MasterType.c_dict[c]

#            ONNX str |    c      |   numpy  |            min                   |              max
TYPE_MAP = [("FLOAT16","float16_t",np.float16,"float16_t.make_infinity(0)"      ,"float16_t.make_infinity(1)"),
            ("FLOAT"  ,"float"    ,np.float32,"cast<float  >(Expr(-FLT_MAX))"   ,"cast<float  >(Expr(FLT_MAX))"),
            ("DOUBLE" ,"double"   ,np.float64,"cast<double >(Expr(-DBL_MAX))"   ,"cast<double >(Expr(DBL_MAX))"),
            ("BOOL"   ,"int8_t"   ,np.bool   ,"cast<int8_t >(Expr(-CHAR_MAX))"  ,"cast<int8_t >(Expr(CHAR_MAX))"),
            ("INT32"  ,"int32_t"  ,np.int32  ,"cast<int32_t>(Expr(-INT_MAX))"   ,"cast<int32_t>(Expr(INT_MAX))"),
            ("INT64"  ,"int64_t"  ,np.int64  ,"cast<int64_t>(Expr(-LLONG_MAX))" ,"cast<int64_t>(Expr(LLONG_MAX))")]

for ts in TYPE_MAP:
    MasterType(*ts)

def from_onnx_t(onnx_t: int) -> MasterType:
    return MasterType.onnx_int_dict[onnx_t]

class VI:
    def __init__(self, value_info: TypeProto) -> None:
        self.tensor_type = value_info.tensor_type

    @property
    def shape(self) -> List[int]:
        return [d.dim_value for d in self.tensor_type.shape.dim]

    @property
    def t(self) -> MasterType:
        return from_onnx_t(self.tensor_type.elem_type)

    
