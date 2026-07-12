#!/bin/bash

set -euxo pipefail

# Assuming pybind11 is installed via pip, not tested on conda
PYBIND_PATH=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

cd diffadmm_cpp && mkdir -p build && cd build && rm -rf ./*

cmake .. -Dpybind11_DIR=$PYBIND_PATH
make -j$(nproc)

SO_NAME=$(basename $(find . -name "diffadmm.*.so" | head -n 1))
mv ./${SO_NAME} ../../src/py/diffadmm/${SO_NAME}