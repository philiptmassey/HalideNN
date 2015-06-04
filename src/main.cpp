#include <Halide.h>
#include "utils.h"

int main(int argc, char **argv) {
    Halide::Image<float> test = load_float_image("gray.png");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << test(j,i) << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
