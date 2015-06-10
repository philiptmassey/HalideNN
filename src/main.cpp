#include <Halide.h>
#include "utils.h"
#include "layers.h"
using Halide::Image;
#include "image_io.h"

int main(int argc, char **argv) {

    // Load filenames
    std::string weight0_filenames[2];
    weight0_filenames[0] = "../res/lenet-2/w0-0.png";
    weight0_filenames[1] = "../res/lenet-2/w0-1.png";
    std::string weight1_filenames[4];
    weight1_filenames[0] = "../res/lenet-2/w1-0.png"; weight1_filenames[1] = "../res/lenet-2/w1-1.png";
    weight1_filenames[2] = "../res/lenet-2/w1-2.png";
    weight1_filenames[3] = "../res/lenet-2/w1-3.png";
    std::string weight2_filename = "../res/lenet-2/w2.png";
    std::string weight3_filename = "../res/lenet-2/w3.png";

    std::string bias0_filename = "../res/lenet-2/b0.png";
    std::string bias1_filename = "../res/lenet-2/b1.png";
    std::string bias2_filename = "../res/lenet-2/b2.png";
    std::string bias3_filename = "../res/lenet-2/b3.png";

    // Load images
    Halide::Image<float> weight0_image = load_convolution_weights(
        weight0_filenames, 1, 2, 5);
    Halide::Image<float> weight1_image = load_convolution_weights(
        weight1_filenames, 2, 2, 5);
    Halide::Image<float> weight2_image = load_float_image(weight2_filename);
    Halide::Image<float> weight3_image = load_float_image(weight3_filename);

    Halide::Image<float> bias0_image = load_float_image(bias0_filename);
    Halide::Image<float> bias1_image = load_float_image(bias1_filename);
    Halide::Image<float> bias2_image = load_float_image(bias2_filename);
    Halide::Image<float> bias3_image = load_float_image(bias3_filename);

    // Load functions
    Halide::Var x, y, z, i;
    Halide::Func weight0, weight1, weight2, weight3;
    Halide::Func bias0, bias1, bias2, bias3;

    weight0(x, y, z, i) = weight0_image(x, y, z, i);
    weight1(x, y, z, i) = weight1_image(x, y, z, i);
    weight2(x, y) = weight2_image(x, y);
    weight3(x, y) = weight3_image(x, y);

    bias0(x, y) = bias0_image(x, y);
    bias1(x, y) = bias1_image(x, y);
    bias2(x, y) = bias2_image(x, y);
    bias3(x, y) = bias3_image(x, y);

    Halide::Image<uint8_t> input(28, 28, 1, 1);
    Halide::Image<uint8_t> input1 = load<uint8_t>("0.png");
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            input(i, j, 0, 0) = input1(i, j);
        }
    }

    Halide::Expr casted = Halide::cast<float>(input(x, y, z, i)) / 255.0f;
    Halide::Func layer0;
    layer0(x, y, z, i) = casted;

    Halide::Func layer1 = convolution_layer(layer0, weight0, bias0, 1, 5, 2);
    Halide::Func layer2 = convolution_layer(layer1, weight1, bias1, 2, 5, 2);
    Halide::Func flattened = flatten(layer2, 4);
    Halide::Func layer3 = fully_connected_layer(flattened, weight2, bias2, 32);
    Halide::Func layer4 = fully_connected_layer(layer3, weight3, bias3, 2);
    Halide::Func layer5 = softmax_layer(layer4, 10);

    Halide::Image<int> output(1, 1, 1);
    layer5.realize(output);
    printf("%d\n", output(0, 0, 0));

    return 0;
}
