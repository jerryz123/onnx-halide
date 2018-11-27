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
        HalogenType.c_dict[self.c] = self

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


class HalideObj:
    def __init__(self, name=None, shape=-1, type=None, io=0):
        self._name = name
        self._shape = -1 if shape == -1 else \
                      tuple([int(i) for i in shape])
        self._type = type
        self._io = io

    @property
    def name(self):
        assert(self._name)
        return self._name
    @property
    def shape(self):
        assert(self._shape != -1)
        return list(self._shape)
    @property
    def size(self):
        return int(np.prod(self.shape))
    @property
    def dims(self):
        return len(self.shape)
    @property
    def is_scalar(self):
        assert(self._shape != -1)
        return self._shape == ()
    @property
    def type(self):
        assert(self._type)
        return self._type
    @property
    def is_input(self):
        assert(self._io != 0)
        return self._io == 1
    @property
    def is_output(self):
        assert(self._io != 0)
        return self._io == -1

    def set_shape(self, shape):
        assert(all([type(i) == int for i in shape]))

        if (self._shape != tuple(shape) and self._shape != -1):
            print(self._shape, tuple(shape))
            assert(False)
        self._shape = tuple(shape)
    def set_type(self, typ):
        assert(not self._type or self._type == typ)
        assert(type(typ) == HalogenType)
        self._type = typ

    def __str__(self):
        return "({}, {}, {}, {})".format(self._name,
                                         self._shape,
                                         self._type.c,
                                         {1:"INPUT",
                                          0:"NODE",
                                          -1:"OUTPUT"}[self._io])
