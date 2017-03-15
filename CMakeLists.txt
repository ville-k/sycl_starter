cmake_minimum_required(VERSION 3.2.2)
project(sycl_testing)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(FindOpenCL)
include(FindComputeCpp)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(SYSTEM ${COMPUTECPP_INCLUDE_DIRECTORY})
include_directories(SYSTEM ${OpenCL_INCLUDE_DIR})


add_executable(main_sycl ${CMAKE_CURRENT_SOURCE_DIR}/main_sycl.cc)
target_compile_options(main_sycl PUBLIC -std=c++11 -Wall)
add_sycl_to_target(main_sycl ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/main_sycl.cc)


add_executable(main_eigen ${CMAKE_CURRENT_SOURCE_DIR}/main_eigen.cc)
target_compile_options(main_eigen PUBLIC -std=c++11 -Wall)
add_sycl_to_target(main_eigen ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/main_eigen.cc)

