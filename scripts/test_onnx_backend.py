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
# tests = [r'test_abs_cpu',
#          r'test_abs_cpu',
#          r'test_acos_cpu',
#          r'test_acos_example_cpu',
#          r'test_add_bcast_cpu',
#          r'test_add_cpu',
#          r'test_and2d_cpu',
#          r'test_and3d_cpu',
#          r'test_and4d_cpu',
#          r'test_and_bcast3v1d_cpu',
#          r'test_and_bcast3v2d_cpu',
#          r'test_and_bcast4v2d_cpu',
#          r'test_and_bcast4v3d_cpu',
#          r'test_and_bcast4v4d_cpu',
#          r'test_argmax_default_axis_example_cpu',
#          r'test_argmax_default_axis_random_cpu',
#          r'test_argmax_keepdims_example_cpu',
#          r'test_argmax_keepdims_random_cpu',
#          r'test_argmax_no_keepdims_example_cpu',
#          r'test_argmax_no_keepdims_random_cpu',
#          r'test_argmin_default_axis_example_cpu',
#          r'test_argmin_default_axis_random_cpu',
#          r'test_argmin_keepdims_example_cpu',
#          r'test_argmin_keepdims_random_cpu',
#          r'test_argmin_no_keepdims_example_cpu',
#          r'test_argmin_no_keepdims_random_cpu',
#          r'test_asin_cpu',
#          r'test_asin_example_cpu',
# ]
# for t in tests:
#   backend_test.exclude(t)
  # import all test cases at global scope to make them visible to python.unittest
backend_test.exclude(r'test_a[a-t][a-z,_]*')

globals().update(backend_test.enable_report().test_cases)
if __name__ == '__main__':




  unittest.main(verbosity=2, failfast=True)
