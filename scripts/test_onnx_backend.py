from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest
import onnx.backend.test

from onnx_halide.backend import HalideBackend


# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(HalideBackend, __name__)

# No support for recurrent operators yet
backend_test.exclude(r'test_(operator_|)gru_[a-z,_]*')
backend_test.exclude(r'test_(operator_|)lstm_[a-z,_]*')
backend_test.exclude(r'test_(operator_|)mvn_[a-z,_]*')
backend_test.exclude(r'test_[a-z,_]*rnn_[a-z,_]*')
# No support for reshaping yet
backend_test.exclude(r'test_(operator_|)expand_[a-z,_]*')
backend_test.exclude(r'test_(operator_|)tile_[a-z,_]*')
backend_test.exclude(r'test_reshape_[a-z,_]*')
backend_test.exclude(r'test_PixelShuffle[a-z,_]*')

# TODO support these
backend_test.exclude(r'test_(operator_|)top_k_[a-z,_]*')
backend_test.exclude(r'test_(operator_|)upsample[a-z,_]*')
backend_test.exclude(r'test_operator_pow[a-z,_]*') # TODO support proper NaNs
backend_test.exclude(r'test_operator_repeat[a-z,_]*') # TODO support proper NaNs

# backend_test.exclude(r'test_bvlc_alexnet_cpu')
# backend_test.exclude(r'test_densenet121_cpu')
# backend_test.exclude(r'test_inception_v1_cpu')
# backend_test.exclude(r'test_inception_v2_cpu')
# backend_test.exclude(r'test_resnet50_cpu')
# backend_test.exclude(r'test_shufflenet_cpu')
# backend_test.exclude(r'test_squeezenet_cpu')
# backend_test.exclude(r'test_vgg19_cpu')
# backend_test.exclude(r'test_zfnet512_cpu')

tests = [
  "OnnxBackendNodeModelTest",
  "OnnxBackendPyTorchConvertedModelTest",
  "OnnxBackendPyTorchOperatorModelTest",
  "OnnxBackendSimpleModelTest",
  "OnnxBackendRealModelTest",
]
for t in tests:
  globals().update({t: backend_test.enable_report().test_cases[t]})
if __name__ == '__main__':
  unittest.main(verbosity=5, failfast=True)

