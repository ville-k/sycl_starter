cmake_minimum_required(VERSION 3.2.2)
project(sycl_testing)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(FindOpenCL)
include(FindComputeCpp)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(SYSTEM ${OpenCL_INCLUDE_DIR})

option(TRISYCL_INCLUDE_DIR "Path to triSYCL include directory")

add_executable(main_sycl ${CMAKE_CURRENT_SOURCE_DIR}/main_sycl.cc)
target_compile_options(main_sycl PUBLIC -std=c++11 -Wall)
target_include_directories(main_sycl PUBLIC
        "${COMPUTECPP_INCLUDE_DIRECTORY}"
)
add_sycl_to_target(main_sycl ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/main_sycl.cc)


# build against triSYCL https://github.com/triSYCL/triSYCL
if(TRISYCL_INCLUDE_DIR)
    add_executable(main_trisycl ${CMAKE_CURRENT_SOURCE_DIR}/main_sycl.cc)
    target_include_directories(main_trisycl PUBLIC
        "${TRISYCL_INCLUDE_DIR}"
        "/usr/include/compute/"
    )
    target_link_libraries(main_trisycl pthread)
    target_compile_options(main_trisycl PUBLIC -std=c++14 -Wall)
endif()


add_executable(main_eigen ${CMAKE_CURRENT_SOURCE_DIR}/main_eigen.cc)
target_compile_options(main_eigen PUBLIC -std=c++11 -Wall)
target_include_directories(main_eigen PUBLIC
        "${COMPUTECPP_INCLUDE_DIRECTORY}"
)
add_sycl_to_target(main_eigen ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/main_eigen.cc)


if(TRISYCL_INCLUDE_DIR)
    add_executable(trisycl_eigen ${CMAKE_CURRENT_SOURCE_DIR}/main_eigen.cc)
    target_include_directories(trisycl_eigen PUBLIC
            "${TRISYCL_INCLUDE_DIR}"
            "/usr/include/compute/"
            )
    target_link_libraries(trisycl_eigen pthread)
    target_compile_options(trisycl_eigen PUBLIC -std=c++14 -Wall)
endif()
