#include <iostream>
#include <algorithm>

#include <CL/sycl.hpp>

// Ensure Eigen::Tensor uses the SYCL backend
#define EIGEN_USE_SYCL 1
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace cl;
using namespace Eigen;



int main(int argc, char **argv) {
    sycl::default_selector selector;
    //sycl::host_selector selector;

    Eigen::QueueInterface queueInterface(selector);
    // Use CPU device to ease debugging
    //Eigen::DefaultDevice device;
    Eigen::SyclDevice device(&queueInterface);

    const int64_t ROWS = 2;
    const int64_t COLS = 2;

    const std::array<int64_t, 2> tensorRange = {{ROWS, COLS}};
    const size_t dataSize = accumulate(tensorRange.begin(), tensorRange.end(), 0) * sizeof(float);

    float *data = static_cast<float *>(device.allocate(dataSize));
    TensorMap<Tensor<float, 2>> a(data, tensorRange);

    float *data2 = static_cast<float *>(device.allocate(dataSize));
    TensorMap<Tensor<float, 2>> b(data2, tensorRange);

    a.device(device) = a.constant(102) + b.constant(201);

    // Eigen device manages buffers - the pointer returned from device.allocate
    // is just a handle that can't be directly used
    //sycl::buffer<uint8_t , 1>
    auto buffer = device.get_sycl_buffer(data);
    cout << "buffer size: " << buffer.get_size() << endl;
    queueInterface.sycl_queue().submit([&](sycl::handler &cgh) {
        auto access = buffer.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for< class test_kernel >(
            sycl::nd_range<1>(sycl::range<1>(4), sycl::range<1>(2)),
            [=](sycl::nd_item<1> item) {
                size_t thread_id = item.get_global(0);
                // size_t total_threads = item.get_global_range()[0];

                // cast byte buffer to actual type
                float* data = ConvertToActualTypeSycl(float, access);
                data[thread_id] += 1;
            });
    });

    device.memcpyDeviceToHost(a.data(), data, dataSize);

    cout << a << endl;

    return 0;
}

