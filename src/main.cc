#include "network.tcc"
#include "layer.h"
#include "tensor.h"

#include <iostream>
#include <bitset>
#include <fstream>

size_t const IMAGE_MAGIC = 2051;
size_t const LABEL_MAGIC = 2049;
size_t const NUM_PIXELS = 784;


namespace RANDOM
{
    std::random_device dev;
    size_t SEED = dev();
    namespace BOUNDS
    {
        double UPPER = 0.08;
        double LOWER = -0.08;
    }
}

template<typename T, size_t Bytes=sizeof(T)>
T read_byte(std::istream & file)
{
    T data{};
    for (size_t i{}; i < Bytes; ++i)
    {
        file.read(reinterpret_cast<char*>(&data), 1);
        data <<= 8;
    }
    return data >>= 8;
}

template<typename T, size_t Pixels>
Tensor<T> read_image(std::istream & file)
{
    std::array<u_int8_t, Pixels> raw_data;
    file.read(reinterpret_cast<char*>(raw_data.data()), Pixels);
    std::vector<double> data(Pixels);

    std::transform(begin(raw_data), end(raw_data), begin(data), [](auto val){return val/255.0;});

    return Tensor<T> {1, Pixels, data};
}

template<typename T>
Tensor<T> read_label(std::istream & file)
{
    u_int8_t data;
    file.read(reinterpret_cast<char*>(&data), 1);
    Tensor<T> out(1, 10);
    out.at(0, data) = 1;
    return out;
}

template<typename T>
bool same_max_index(Tensor<T> const& lhs, Tensor<T> const& rhs)
{
    assert(lhs.shape() == rhs.shape());
 
    std::pair<size_t, size_t> rhs_index{0, 0}, lhs_index{0, 0};
    for (size_t i{}; i < lhs.shape().first; ++i)
    {
        for (size_t j{1}; j < lhs.shape().second; ++j)
        {
            if (lhs.at(lhs_index.first, lhs_index.second) < lhs.at(i, j))
            {
                lhs_index = {i, j};
            }

            if (rhs.at(rhs_index.first, rhs_index.second) < rhs.at(i, j))
            {
                rhs_index = {i, j};
            }
        }
    }

    return rhs_index == lhs_index;
}

int main()
{
    std::ifstream image_stream("/home/nisse/projects/the_numbers_mason/archive/train-images-idx3-ubyte/train-images-idx3-ubyte", std::ios::binary);
    std::ifstream label_stream("/home/nisse/projects/the_numbers_mason/archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte", std::ios::binary);

    u_int32_t image_magic{read_byte<u_int32_t>(image_stream)};
    u_int32_t num_images{read_byte<u_int32_t>(image_stream)};
    u_int32_t rows{read_byte<u_int32_t>(image_stream)};
    u_int32_t cols{read_byte<u_int32_t>(image_stream)};

    u_int32_t label_magic{read_byte<u_int32_t>(label_stream)};
    u_int32_t num_labels{read_byte<u_int32_t>(label_stream)};
    
    assert(image_magic == IMAGE_MAGIC);
    assert(label_magic == LABEL_MAGIC);
    assert(rows * cols == NUM_PIXELS);
    assert(num_images == num_labels);

    Network<double> net(0.01,
    LAYER::Linnear(NUM_PIXELS, 128),
    LAYER::Relu(),
    LAYER::Linnear(128, 10),
    LAYER::Softmax());

    std::cout << "TRAINING ON " << num_images << " IMAGES" << std::endl;
    for (size_t image{}; image < num_images; ++image)
    {
        auto X = read_image<double, NUM_PIXELS>(image_stream);
        auto Y = read_label<double>(label_stream);
        net.train(1, X, Y);
    }
    
    std::ifstream test_image_stream("archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", std::ios::binary);
    std::ifstream test_label_stream("archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte", std::ios::binary);
    
    u_int32_t test_image_magic{read_byte<u_int32_t>(test_image_stream)};
    u_int32_t test_num_images{read_byte<u_int32_t>(test_image_stream)};
    u_int32_t test_rows{read_byte<u_int32_t>(test_image_stream)};
    u_int32_t test_cols{read_byte<u_int32_t>(test_image_stream)};
    
    u_int32_t test_label_magic{read_byte<u_int32_t>(test_label_stream)};
    u_int32_t test_num_labels{read_byte<u_int32_t>(test_label_stream)};
    
    assert(test_image_magic == IMAGE_MAGIC);
    assert(test_label_magic == LABEL_MAGIC);
    assert(test_rows * test_cols == NUM_PIXELS);
    assert(test_num_images == test_num_labels);

    std::cout << "TESTING ON " << test_num_images << " IMAGES" << std::endl;
    int correct{};
    for (size_t image{}; image < test_num_images; ++image)
    {
        auto X = read_image<double, NUM_PIXELS>(test_image_stream);
        auto Y = read_label<double>(test_label_stream);
        correct += same_max_index(Y, net.predict(X));
    }

    std::cout << "RESULT:" << std::endl;
    std::cout << correct << "/" << test_num_images << std::endl;
    std::cout << static_cast<double>(correct)/test_num_images * 100 << "%" << std::endl;

    return 0;
}
