from setuptools import setup
import subprocess

subprocess.run("cd runtime && ./build-runtime.sh", shell=True)

setup(name="onnx-halide")
