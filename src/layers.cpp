#include <math.h>

#include <Halide.h>

Halide::Func convolution_layer(Halide::Func input, Halide::Func W,
    Halide::Func b, int num_input_nodes, int filter_size, int pool_size) {
    
    Halide::Func convolution, pool, biased;
    Halide::Var x, y, z, i;
    Halide::RDom conv_r(0, filter_size, 0, filter_size, 0, num_input_nodes);
    Halide::RDom pool_r(0, pool_size, 0, pool_size);

    convolution(x, y, z, i) = 0.0f;
    convolution(x, y, z, i) += W(conv_r.x, conv_r.y, conv_r.z, z) *
        input(x + conv_r.x, y + conv_r.y, conv_r.z, i);

    pool(x, y, z, i) = -1000000.0f;
    pool(x, y, z, i) = Halide::max(convolution(pool_size * x + pool_r.x,
        pool_size * y + pool_r.y, z, i), pool(x, y, z, i));

    biased(x, y, z, i) = tanh(pool(x, y, z, i) + b(z, 0));

    //Halide::Var x_inner, x_outer, y_inner, y_outer;
    //biased.parallel(i);
    //biased.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 2);
    //biased.vectorize(x_inner);
    //biased.unroll(y_inner);

    return biased;
}

Halide::Func flatten(Halide::Func input, int image_size) {

    Halide::Func flatten1, flatten2;
    Halide::Var x, y, i;

    flatten1(x, y, i) = input(x / image_size, x % image_size, y, i);

    int full_size = image_size * image_size;
    flatten2(x, y, i) = 0.0f;
    flatten2(x, 0, i) = flatten1(x % full_size, x / full_size, i);

    //flatten2.parallel(i);
    //flatten2.vectorize(x, 4);

    return flatten2;
}

Halide::Func fully_connected_layer(Halide::Func input, Halide::Func W,
    Halide::Func b, int num_input_nodes) {

    // This function assumes the input layer has been flattened.
    
    Halide::Func product;
    Halide::Var x, y, i;
    Halide::RDom r(0, num_input_nodes);

    product(x, y, i) = 0.0f;
    product(x, 0, i) += W(x, r.x) * input(r.x, 0, i);
    product(x, 0, i) = tanh(product(x, 0, i) + b(x, 0));

    //product.vectorize(x, 4);

    return product;
}

Halide::Func softmax_layer(Halide::Func input, int num_input_nodes) {

    Halide::Func transform, softmax;
    Halide::Var x, y, i;
    Halide::RDom r(0, num_input_nodes);

    transform(x, y, i) = exp(input(x, y, i));
    softmax(x, y, i) = Halide::argmax(transform(r.x, 0, i))[0];

    return softmax;
}
