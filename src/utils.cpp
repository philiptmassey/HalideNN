#include <Halide.h>

using Halide::Image;
#include "image_io.h"

#define FLOAT_OFFSET 1000000.0f;

Halide::Image<float> load_float_image(std::string filename) {
    Halide::Var x, y, c;
    Halide::Func expanded, flattened, casted;
    Halide::Image<uint8_t> input = load<uint8_t>(filename);

    expanded(x, y, c) = cast<uint32_t>(input(x, y, c));

    flattened(x, y) = expanded(x, y, 0) << 24;
    flattened(x, y) += expanded(x, y, 1) << 16;
    flattened(x, y) += expanded(x, y, 2) << 8;
    flattened(x, y) += expanded(x, y, 3);

    casted(x, y) = cast<float>(flattened(x, y)) / FLOAT_OFFSET;

    Halide::Image<float> output(input.width(), input.height());
    casted.realize(output);
    return output;
}
