# onnx-halide: A Halide backend for ONNX
This tool converts ONNX neural networks to Halide generators. 

## Installation
This tool has three dependencies.
`numpy` and `onnx` can be fetched from PyPI.
``` 
pip3 install numpy onnx
git clone https://github.com/jerryz123/onnx-halide.git
cd onnx-halide
pip3 install -e .
```
Halide can be build from source, or fetched from a nightly build. 
Either way, `HALIDE_ROOT_DIR` should point to the install directory for Halide.

### Installing Halide from source
Follow instructions at [https://github.com/halide/Halide](https://github.com/halide/Halide)

### Installing Halide from nightly
Fetch and extract correct version from [https://buildbot.halide-lang.org/](https://buildbot.halide-lang.org/)

Set `HALIDE_ROOT_DIR` to destination directory.

### Installing Caffe2
To run comparisons against a reference Caffe2 backend, [install PyTorch+Caffe2](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=prebuilt)

## Testing
This tool can be verified using the ONNX backend regression suite.
Most operators, and all full-model tests pass.

`python3 scripts/test_onnx_backend.py`

## Usage
This tool follows the standard usage convention for ONNX backends.
Load an onnx model with `onnx_halide.backend.prepare`.
```
import onnx
import onnx_halide.backend as backend

model = onnx.load("model.onnx")
prepared = backend.prepare(model)
```
The bulk of the work performed by the tool is in `backend.prepare`.
Intermediate files are placed in a `generated/` directory. 
The order of operations performed in this function is as follows.

1. Preprocess the ONNX operator graph, performing shape and type inferencing for all intermediate tensors.
2. Generate source code for a Halide generator that represents the ONNX graph. This will appear as `halogen_generator.cpp`. An example of what this code looks like can be seen [here](https://gist.github.com/jerryz123/525336f72aedea651a5e91f0a2fbd021)
3. Compile `halogen_generator.cpp` with `libHalide.a` to output a Halide generator binary, `halogen.generator`
4. Execute `halogen.generator`, targeting CPU, and generate `halogen.a` and `halogen.h`. These contain the actual executable code for the neural network
5. Generate a C interface function to the generated pipeline as `halogen_c.cpp`, and link with `halogen.a` to produce a shared object file `lib<model_name>.so`.
6. Link `lib<model_name>.so` into Python using `ctypes`, to provide a Python interface to the compiled pipeline.

At this point `prepared.run` will accept inputs to the model as Numpy arrays, execute the model on these inputs, and return the output as Numpy arrays.
```
results = prepared.run(inputs)
```

An example script to show complete usage of this tool on models from the [ONNX model zoo](https://github.com/onnx/models) is at `scripts/test_models.py`.

### Intermediate files
While it is possible to use this tool end-to-end in only Python, the intermediate files of the tool may be useful for certain applications.

`halogen_generator.cpp` contains human-readable source code for a Halide generator, with comments and annotations indicating the operators of the model, their inputs, and the results of shape and type inference. While the end-to-end system just compiles this file, the source code for the generator may be useful for other applications, such as hardware synthesis or autoscheduling.

`halogen.generator` is an executable Halide generator. The end-to-end system executes this binary with the `target=host-no_asserts -e h,static_library` flags. Executables can be generated for other supported targets manually by changing the `target` flag, and the generated code can be emitted as other forms by changing the `-e` flag. Note that `no_asserts` is necessary due to poor runtime of assertion generation in Halide.

`halogen.a` and `halogen.h` is the static library with the generated Halide pipeline output by `halogen.generator`. The end-to-end system compiles these into a shared library so they can be linked into Python. However, the library can be linked with by any C code, and the generated function as defined in `halogen.h` can be called.

## Design
### Procedure
### Operator Generators
### Type and Shape Inference
### Algorithm Generation
### Schedule Generation

## To-do
