# SYCL & Eigen Starter Project

## Prerequisites

* SYCL implementation:
  * Eigen starter requires [ComputeCPP](https://www.codeplay.com/products/computesuite/computecpp) from Codeplay
  * SYCL starter works with [ComputeCPP](https://www.codeplay.com/products/computesuite/computecpp) and [triSYCL](https://github.com/triSYCL/triSYCL)
* Eigen linear algebra library with SYCL support. Bleeding edge SYCL development is [happening here](https://bitbucket.org/benoitsteiner/opencl)

## Contents

* main_sycl.cc   - share template code between C++ and OpenCL/SYCL
* main_eigen.cc  - execute tensor operations on OpenCL/SYCL device

## Building

        mkdir build
        cd build
        cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen/source/root/ \
              -DTRISYCL_INCLUDE_DIR=/path/to/trisycl/root/include \
              -DCOMPUTECPP_PACKAGE_ROOT_DIR=/path/to/computecpp/sdk/root/ ..
        cmake --build .
