import numpy as np
import ctypes
from onnx import TensorProto



class HalogenType:
    onnx_int_dict = {}
    c_dict = {}
    def __init__(self, onnx_str, c, numpy, ct, min, max):
        self.onnx_int    = {k:v for k,v in TensorProto.DataType.items()}[onnx_str]
        self.onnx_str    = onnx_str
        self.c           = c
        self.np          = numpy
        self.ct          = ct
        self.ct_ptr      = ctypes.POINTER(ct)
        self.c_min       = min
        self.c_max       = max
        HalogenType.onnx_int_dict[self.onnx_int] = self
        HalogenType.c_dict       [self.c] = self

    def __eq__(self, other):
        return self.onnx_int == other.onnx_int

    def from_onnx(onnx_int):
        return HalogenType.onnx_int_dict[onnx_int]
    def from_c(c):
        return HalogenType.c_dict[c]

#            ONNX str |    c      |   numpy  |    ctypes       |            min                   |              max
TYPE_MAP = [("FLOAT16","float16_t",np.float16,ctypes.c_short   ,"float16_t.make_infinity(0)"      ,"float16_t.make_infinity(1)"),
            ("FLOAT"  ,"float"    ,np.float32,ctypes.c_float   ,"cast<float  >(Expr(-FLT_MAX))"   ,"cast<float  >(Expr(FLT_MAX))"),
            ("DOUBLE" ,"double"   ,np.float64,ctypes.c_double  ,"cast<double >(Expr(-DBL_MAX))"   ,"cast<double >(Expr(DBL_MAX))"),
            ("BOOL"   ,"int8_t"   ,np.bool   ,ctypes.c_char    ,"cast<int8_t >(Expr(-CHAR_MAX))"  ,"cast<int8_t >(Expr(CHAR_MAX))"),
            ("INT32"  ,"int32_t"  ,np.int32  ,ctypes.c_int     ,"cast<int32_t>(Expr(-INT_MAX))"   ,"cast<int32_t>(Expr(INT_MAX))"),
            ("INT64"  ,"int64_t"  ,np.int64  ,ctypes.c_longlong,"cast<int64_t>(Expr(-LLONG_MAX))" ,"cast<int64_t>(Expr(LLONG_MAX))")]

for ts in TYPE_MAP:
    HalogenType(*ts)

    
