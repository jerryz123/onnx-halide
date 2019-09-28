#!/usr/bin/env bash

set -e
set -x
set -v
set -o pipefail

g++ -O3 -std=c++11 -I $RISCV/include/ -I $RISCV/share/halide/tools/ \
    -Wall -Werror -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -fno-rtti \
    ./runtime_generator.c $RISCV/lib/libHalide.so $RISCV/share/halide/tools/GenGen.cpp \
    -o generator.bin  -ldl -lpthread -lz -lz -lrt -ldl -ltinfo -lpthread -lm -lxml2

./generator.bin -r HalideRuntime -e o,h -o ./ target=riscv-64-linux-no_asserts-no_bounds_query
