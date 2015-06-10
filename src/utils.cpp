#include <Halide.h>

using Halide::Image;
#include "image_io.h"

#define FLOAT_OFFSET 10000000.0f;

Halide::Image<float> load_float_image(std::string filename) {
    
    Halide::Var x, y, c;
    Halide::Func expanded, flattened, casted;
    Halide::Image<uint8_t> input = load<uint8_t>(filename);

    expanded(x, y, c) = Halide::cast<uint32_t>(input(x, y, c));

    flattened(x, y) = expanded(x, y, 0) << 24;
    flattened(x, y) += expanded(x, y, 1) << 16;
    flattened(x, y) += expanded(x, y, 2) << 8;
    flattened(x, y) += expanded(x, y, 3);

    casted(x, y) = Halide::cast<float>(flattened(x, y)) / FLOAT_OFFSET;
    casted(x, y) = Halide::select(expanded(x, y, 3) % 2 == 0,
        casted(x, y) * -1.0f, casted(x, y));

    Halide::Image<float> output(input.width(), input.height());
    casted.realize(output);
    return output;
}

Halide::Image<float> load_convolution_weights(std::string* filenames, 
    int num_input_nodes, int num_output_nodes, int filter_size) {

    Halide::Image<float> weights(filter_size, filter_size, num_input_nodes,
        num_output_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        for (int j = 0; j < num_output_nodes; j++) {
            std::string filename = filenames[num_input_nodes * i + j];
            Halide::Image<float> weight = load_float_image(filename);
            for (int x = 0; x < filter_size; x++) {
                for (int y = 0; y < filter_size; y++) {
                    weights(x, y, i, j) = weight(x, y);
                }
            }
        }
    }
    return weights;
}
