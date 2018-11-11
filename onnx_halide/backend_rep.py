from onnx.backend.base import BackendRep
from onnx import TensorProto
import subprocess
import numpy as np

HALIDE_DIR = "/home/jerry/Projects/Halide"

ONNX_TYPE_DECODER = lambda x:{k: v for (v, k) in TensorProto.DataType.items()}[x]
C_TYPE_DECODER = lambda x: {"FLOAT" : "float"}\
                 [ONNX_TYPE_DECODER(x)]

class HalideBackendRep(BackendRep):
    def __init__(self, model):

        self.halide_str = """"""
        self.indent = 0
        self.buffers = {}
        self.model_name = "{}_{}_{}".format(model.graph.name, model.model_version,
                                       model.domain.replace('.', '-'))
        self.generate_csrc(model)

    def cpp(self, s="", incr=0):
        if incr < 0:
            self.indent += incr
        self.halide_str += "{}{}\n".format("  " * self.indent, s)
        if incr > 0:
            self.indent += incr

    def generate_csrc(self, model):
        self.cpp("#include \"Halide.h\"")
        self.cpp("using namespace Halide;")
        self.cpp()

        init_data = {}
        # Find initial values
        for init in model.graph.initializer:
            if init.raw_data:
                onnx_data = np.frombuffer(init.raw_data, dtype=float)
            c_arr = ", ".join([str(i) for i in onnx_data])
            init_data[init.name] = c_arr

        # Create arrays for input buffers, assign these with initial vlaues
        for ip in model.graph.input:
            c_shape = [d.dim_value for d in ip.type.tensor_type.shape.dim]
            c_type  = C_TYPE_DECODER(ip.type.tensor_type.elem_type)
            c_name  = ip.name.replace('/', '')
            c_size  = np.prod(c_shape)
            c_str   = "{} c_{}[{}]".format(c_type, c_name, c_size)
            if ip.name in init_data:
                c_str += " = {{{}}}".format(init_data[ip.name])
            c_str += ";"
            self.cpp(c_str)
            self.buffers[ip.name] = (c_name, c_shape, c_type)

        # Create arrays for output buffers
        for op in model.graph.output:
            c_type = C_TYPE_DECODER(op.type.tensor_type.elem_type)
            c_name = op.name.replace('/', '')
            c_size = np.prod([d.dim_value for d in op.type.tensor_type.shape.dim])
            self.cpp("{} c_{}[{}];".format(c_type, c_name, c_size))

        # Create arrays for constant nodes
        for cnode in model.graph.node:
            if cnode.op_type == "Constant":
                for (op_name, attr) in zip(cnode.output, cnode.attribute):
                    c_name  = op_name.replace('/', '')
                    c_type  = C_TYPE_DECODER(attr.t.data_type)
                    c_shape = [d for d in attr.t.dims]
                    c_size  = np.prod(c_shape)
                    if not c_shape: # Scalar const
                        if attr.t.float_data:
                            self.cpp("{} c_{} = {};".format(c_type, c_name, attr.t.float_data[0]))
                        else:
                            raise NotImplementedError
                    else:
                        if attr.t.float_data:
                            c_arr = ", ".join([str(i) for i in attr.t.float_data])
                        else:
                            raise NotImplementedError
                        self.cpp("{} c_{}[{}] = {{{}}};".format(c_type, c_name, c_size, c_arr))

                    print(op_name, c_name, c_type, c_shape)




        self.cpp()
        self.cpp("void halide_compute() {", 1);

        for n, (c_name, c_shape, c_type) in self.buffers.items():
            self.cpp("Halide::Buffer<{0}> {1}(c_{1}, {2});".format(
                c_type,
                c_name,
                ', '.join([str(i) for i in c_shape])))
        self.cpp()
        for nidx, node in enumerate(model.graph.node):
            self.generate_node(nidx, node)
            break
        self.cpp("};", -1)

        self.cpp("int main(){", 1)
        self.cpp("halide_compute();")
        self.cpp("printf(\"Done\\n\");")
        self.cpp("};", -1)


        with open("halonet.cc", 'w') as f:
            f.write(self.halide_str)
        r = subprocess.run(["llvm-config", "--cxxflags", "--ldflags", "--libs"],
                           stdout=subprocess.PIPE)
        llvm_flags = r.stdout.decode().replace('\n', ' ')
        cmd  = "g++ -g -Wall -xc++ - -ldl "
        cmd += "-o halonet "
        cmd += "{} ".format(llvm_flags)
        cmd += "-lHalide -I{0}/include -L{0}/lib ".format(HALIDE_DIR)
        print(cmd)
        r = subprocess.run(cmd,
                           shell=True,
                           input=self.halide_str.encode())
        exit(0)

    def generate_var(self, var):
        self.cpp("Var {0}(\"{0}\");".format(var))

    def generate_func(self, fname):
        self.cpp("Func {0}(\"{0}\");".format(fname))


    def generate_node(self, nidx, node):
        print(node)
        for op in node.output:
            self.generate_func(op)
        self.cpp("{{ // {} {} {}".format(node.op_type, nidx, node.name), 1)
        # for attr in node.attribute:
        #     onnx_type = ONNX_TYPE_DECODER(attr.type)

        #     for idx, i in enumerate(attr.ints):
        #         self.generate_var_decl("{}_{}".format(attr.name, str(idx)))

        if node.op_type == "Conv":
            op = node.output[0]
            strides, pads, kernel_shape = node.attribute
            ip, weight, bias = node.input
            self.generate_var("n")
            self.generate_var("c")
            self.generate_var("x")
            self.generate_var("y")

            self.cpp("int  n_in = {}.dim(0).extent();".format(ip))
            self.cpp("int ch_in = {}.dim(1).extent();".format(ip))
            self.cpp("int  h_in = {}.dim(2).extent();".format(ip))
            self.cpp("int  w_in = {}.dim(3).extent();".format(ip))
            self.cpp()
            self.generate_func("in_bounded")
            self.cpp("in_bounded = BoundaryConditions::constant_exterior(", 1)
            self.cpp("{}, 0);".format(ip))
            self.cpp("", -1)
            self.cpp("RDom r(0,ch_in, 0,{}, 0,{});".format(kernel_shape.ints[0], kernel_shape.ints[1]))
            self.cpp()
            self.cpp("{}(n,c,x,y) = {}(c);".format(op, bias))
            self.cpp()
            self.cpp("{}(n,c,x,y) += {}(n,r[0],r[1],r[2]) \
* in_bounded(n,r[0],x*{}+r[1],y*{}+r[2]);".format(
    op, weight, strides.ints[0], strides.ints[1]))
        elif node.op_type == "Div":
            print(node)
            
            a = 1
            
        else:
            print(node.op_type)
            raise NotImplementedError
        self.cpp("}", -1)
        self.cpp("{}.realize();".format(node.output[0]))
