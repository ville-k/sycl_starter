#include <iostream>
#include <CL/sycl.hpp>

using namespace std;
using namespace cl::sycl;

template<typename GeneratedType>
struct Generator {
    Generator(size_t start, size_t length)
            : offset_(0), start_(start), length_(length) {

    }

    GeneratedType generate() {
        GeneratedType generated(start_ + offset_);
        offset_ = (offset_ + 1) % length_;
        return generated;
    }

    size_t offset_;
    size_t start_;
    size_t length_;
};

int main(int argc, char **argv) {
    default_selector selector;
    // Use host implementation for easy debugging
    //host_selector selector;

    queue queue(selector);
    const size_t ROWS = 4;
    const size_t COLS = 4;
    float data[ROWS * COLS] = {
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
    };

    Generator<float> host_generator(0, 4);
    for (size_t i = 0; i < 8; ++i) {
        cout << host_generator.generate() << " " << std::endl;
    }

    buffer<float, 2> buffer(data, range<2>(ROWS, COLS));
    queue.submit([&](handler &command_group_handler) {
        auto access = buffer.get_access<access::mode::read_write>(command_group_handler);

        range<2> work_groups(2, 2);
        stream out(4096, 120, command_group_handler);
        command_group_handler.parallel_for_work_group<class matrix_map_g>(
                work_groups,
                [=](group<2> group_id) {

                    auto group_row = group_id.get(0);
                    auto group_col = group_id.get(1);
                    Generator<float> device_generator(0, 4);

                    parallel_for_work_item(group_id, [&](item<2> item_id) {
                        for (size_t row_offset = 0; row_offset < group_id.get_group_range()[0]; ++row_offset) {
                            for (size_t col_offset = 0; col_offset < group_id.get_group_range()[1]; ++col_offset) {
                                size_t matrix_row = group_row * group_id.get_group_range()[0] + row_offset;
                                size_t matrix_col = group_col * group_id.get_group_range()[1] + col_offset;

                                access[matrix_row][matrix_col] = device_generator.generate();
                            }
                        }
                    });
                });
    });

    cout << "Waiting for kernel to finish..." << std::endl;
    queue.wait();

    accessor<float, 2, access::mode::read, access::target::host_buffer> host_accessor(buffer);
    for (size_t i = 0; i < ROWS; i++) {
        for (size_t j = 0; j < COLS; j++) {
            cout << " " << host_accessor[i][j];
        }
        cout << std::endl;
    }
}
