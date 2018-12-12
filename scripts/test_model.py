import numpy as np
import onnx
import onnx.utils
from onnx import helper, TensorProto, shape_inference, numpy_helper

import time
import glob
import os


import caffe2.python.onnx.backend as caffe2_backend
import onnx_halide.backend as backend

network = "bvlc_alexnet"
#network = "inception_v1"
#network = "shufflenet"
#network = "resnet50"
network = "squeezenet"
print("Running network {}".format(network))
model_pb_path = "{}/model.onnx".format(network)
npz_path = "{}/test_data_0.npz".format(network)
test_data_dir = "test_data_set_0"

# Load the model and sample inputs
model = onnx.load(model_pb_path)
try:
    sample = np.load(npz_path, encoding='bytes')
    inputs = list(sample['inputs'])
except:
    inputs = []
    inputs_num = len(glob.glob(os.path.join(network, test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(network, test_data_dir, 'input_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
            inputs.append(numpy_helper.to_array(tensor))


# Run the model with Caffe2 backend
print("Caffe2 preparing")
caffe2_backend = caffe2_backend.prepare(model)
print("Caffe2 starting")
start = time.time()
outputs = caffe2_backend.run(inputs)
end = time.time()
#print("Caffe2 completed ", end - start, " seconds")
print("Caffe2 completed ")


# Run the model with Halide backend
print("Halogen preparing")
halogen_backend = backend.prepare(model)
print("Halogen starting")
start = time.time()
results = halogen_backend.run(inputs)
end = time.time()
#print("Halogen completed ", end - start, " seconds")
print("Halogen completed ")


# Verify results are same
atol = 1e-7
rtol = 0.001
for i, (a, b) in enumerate(zip(results, outputs)):
    fa = a.flatten()
    fb = b.flatten()
    tol = atol + rtol * abs(fb) - abs(fa - fb)
    print("Maximum error: {}".format(np.min(tol)))


