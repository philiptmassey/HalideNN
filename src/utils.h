#include <Halide.h>

Halide::Image<float> load_float_image(std::string filename);
Halide::Image<float> load_convolution_weights(std::string* filenames, 
    int num_input_nodes, int num_output_nodes, int filter_size);
