#include <Halide.h>

Halide::Func convolution_layer(Halide::Func input, Halide::Func W,
    Halide::Func b, int num_input_nodes, int filter_size, int pool_size);
    
Halide::Func flatten(Halide::Func input, int image_size);

Halide::Func fully_connected_layer(Halide::Func input, Halide::Func W,
    Halide::Func b, int num_input_nodes);

Halide::Func softmax_layer(Halide::Func input, int num_input_nodes);
