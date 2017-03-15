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
    device.memcpyDeviceToHost(a.data(), data, dataSize);

    cout << a << endl;

    return 0;
}

