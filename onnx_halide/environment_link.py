import subprocess
import os
from os.path import join
from typing import List, Type

class Environment:
    ''' Provide an interface for making external calls to compilers/linkers.
    Handles setting appropriate environment variables and printing debug output
    on failure.
    '''
    cxx = "g++"
    target_cxx = "riscv64-unknown-linux-gnu-g++"
    target_ld  = "riscv64-unknown-linux-gnu-ld"
    target_arch_triple = "-march=rv64imafdc -mabi=lp64 "
    install_dir = os.environ['RISCV']

    # TODO: Expand class as a means to configure various environment related settings such as
    # compilation target, and set appropriate env vars in subprocess call
    @classmethod
    def run_cmd(cls, cmd: str) -> str:
        try:
            return subprocess.run(cmd, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print("Failed running command\n{}\nReturned with error".format(cmd, e.output))
            raise e

    @classmethod
    def compile_library(cls, src: str, objects: List[str], temp_dir: str) -> str:
        src_cname = join(temp_dir, "generated.c")
        src_oname = join(temp_dir, "generated.o")
        src_aname = join(temp_dir, "generated.a")
        with open(src_cname, 'w') as f:
            f.write(src)

        cmd  = "{} -std=c++11 ".format(cls.target_cxx)
        cmd += "-I./ -fno-rtti "
        cmd += cls.target_arch_triple
        cmd += "-c {} -o {} ".format(src_cname, src_oname)

        r = Environment.run_cmd(cmd)

        cmd  = "riscv64-unknown-linux-gnu-ar rcs {} {}".format(
            src_aname,
            ' '.join([src_oname] + list(objects)))
        r = Environment.run_cmd(cmd)
        return src_aname

    @classmethod
    def compile_raw_to_object(cls, raw_file: str, gen_name: str, temp_dir: str) -> str:
        oname = join(temp_dir, "{}.o".format(gen_name))
        cmd  = "{} --no-relax -r -b binary {} -o {} ".format(cls.target_ld, raw_file, oname)

        r = Environment.run_cmd(cmd)

        return oname

    @classmethod
    def compile_kernel(cls, src: str, gen_name: str, temp_dir: str):
        src_fname = join(temp_dir, "{}.cpp".format(gen_name))
        generator_bin = join(temp_dir, "{}.bin".format(gen_name))
        with open(src_fname, 'w') as f:
            f.write(src)


        cmd  = "{} -std=c++11 ".format(cls.cxx)
        cmd += "-I {} -I {} ".format(join(cls.install_dir, "include"),
                                     join(cls.install_dir, "share/halide/tools"))
        cmd += "-fno-rtti "
        cmd += "{} {} {} ".format(src_fname,
                                  join(cls.install_dir, "lib/libHalide.so"),
                                  join(cls.install_dir, "share/halide/tools/GenGen.cpp"))
        cmd += "-o {} -ldl -lpthread -lz ".format(generator_bin)
        cmd += "-lrt -ldl -ltinfo -lm"

        r = subprocess.run(cmd, check=True, shell=True)

        cmd  = "{} -g {} -o {} ".format(generator_bin, gen_name, temp_dir)
        cmd += "-e h,o "
        cmd += "target=riscv-64-linux-no_asserts-no_runtime-no_bounds_query"

        r = subprocess.run(cmd, check=True, shell=True)

    @classmethod
    def run_model(cls, src: str, library_name: str, temp_dir: str) -> str:
        src_fname = join(temp_dir, "main.c")
        src_bname = join(temp_dir, "main.riscv")
        with open(src_fname, 'w') as f:
            f.write(src)

        cmd  = "{} -std=c++11 ".format(cls.target_cxx)
        cmd += "-fno-rtti -static "
        cmd += cls.target_arch_triple
        cmd += "{} {} -latomic -lpthread -ldl -o {} ".format(src_fname, library_name, src_bname)
        r = Environment.run_cmd(cmd)

        cmd  = "spike pk {}".format(src_bname)
        r = Environment.run_cmd(cmd)
            

