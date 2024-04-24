#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layer.h"
#include "layer.c"
// #include "../../optimizer/optimizer.c"
#include "../../utils/memory.c"
#include "../../utils/optim.c"
#include "../../utils/compute/convolution.c"
#include "../../utils/compute/pooling.c"
#include "../../utils/compute/fully_connected.c"
#include "../../utils/compute/dropout.c"
#include "../../utils/compute/flatten.c"
#include "../../utils/compute/activation.c"
#include "../../utils/rand.c"

int main() {
    // Create a convolutional layer
    LayerParams conv_params;
    conv_params.conv_params.num_filters = 32;
    conv_params.conv_params.filter_size = 3;
    conv_params.conv_params.stride = 1;
    conv_params.conv_params.padding = 1;
    strcpy(conv_params.conv_params.activation, "relu");

    Dimensions input_shape = {28, 28, 1};
    Layer* conv_layer = create_layer(CONVOLUTIONAL, conv_params);
    conv_layer->input_shape = input_shape;
    initialize_layer(conv_layer);
    compute_output_shape(conv_layer);

    printf("Convolutional Layer:\n");
    printf("Input Shape: (%d, %d, %d)\n", conv_layer->input_shape.height, conv_layer->input_shape.width, conv_layer->input_shape.channels);
    printf("Output Shape: (%d, %d, %d)\n", conv_layer->output_shape.height, conv_layer->output_shape.width, conv_layer->output_shape.channels);

    // Create an activation layer after the convolutional layer
    LayerParams activation_params;
    strcpy(activation_params.activation_params.activation, "relu");
    Layer* activation_layer = create_layer(ACTIVATION, activation_params);
    activation_layer->input_shape = conv_layer->output_shape;
    compute_output_shape(activation_layer);

    // Link the layers
    conv_layer->next_layer = activation_layer;
    activation_layer->prev_layer = conv_layer;

    // Create a pooling layer
    LayerParams pool_params;
    pool_params.pooling_params.pool_size = 2;
    pool_params.pooling_params.stride = 2;
    strcpy(pool_params.pooling_params.pool_type, "max");

    Layer* pool_layer = create_layer(POOLING, pool_params);
    pool_layer->input_shape = activation_layer->output_shape;
    compute_output_shape(pool_layer);

    printf("\nPooling Layer:\n");
    printf("Input Shape: (%d, %d, %d)\n", pool_layer->input_shape.height, pool_layer->input_shape.width, pool_layer->input_shape.channels);
    printf("Output Shape: (%d, %d, %d)\n", pool_layer->output_shape.height, pool_layer->output_shape.width, pool_layer->output_shape.channels);

    // Link the layers
    activation_layer->next_layer = pool_layer;
    pool_layer->prev_layer = activation_layer;

    // Create a fully connected layer
    LayerParams fc_params;
    fc_params.fc_params.num_neurons = 10;
    strcpy(fc_params.fc_params.activation, "softmax");

    Layer* fc_layer = create_layer(FULLY_CONNECTED, fc_params);
    fc_layer->input_shape = pool_layer->output_shape;
    initialize_layer(fc_layer);
    compute_output_shape(fc_layer);

    printf("\nFully Connected Layer:\n");
    printf("Input Shape: (%d, %d, %d)\n", fc_layer->input_shape.height, fc_layer->input_shape.width, fc_layer->input_shape.channels);
    printf("Output Shape: (%d, %d, %d)\n", fc_layer->output_shape.height, fc_layer->output_shape.width, fc_layer->output_shape.channels);

    // Link the layers
    pool_layer->next_layer = fc_layer;
    fc_layer->prev_layer = pool_layer;

    // Allocate memory for input and output tensors
    float*** input_tensor = malloc_3d_float_array(input_shape.height, input_shape.width, input_shape.channels);
    float*** conv_output = malloc_3d_float_array(conv_layer->output_shape.height, conv_layer->output_shape.width, conv_layer->output_shape.channels);
    float*** activation_output = malloc_3d_float_array(activation_layer->output_shape.height, activation_layer->output_shape.width, activation_layer->output_shape.channels);
    float*** pool_output = malloc_3d_float_array(pool_layer->output_shape.height, pool_layer->output_shape.width, pool_layer->output_shape.channels);
    float*** fc_output = malloc_3d_float_array(1, 1, fc_layer->output_shape.channels);

    // Forward pass
    layer_forward_pass(conv_layer, input_tensor, conv_output);
    layer_forward_pass(activation_layer, conv_output, activation_output);
    layer_forward_pass(pool_layer, activation_output, pool_output);
    layer_forward_pass(fc_layer, pool_output, fc_output);

    // Allocate memory for gradients
    float*** conv_input_grad = malloc_3d_float_array(input_shape.height, input_shape.width, input_shape.channels);
    float*** activation_input_grad = malloc_3d_float_array(activation_layer->input_shape.height, activation_layer->input_shape.width, activation_layer->input_shape.channels);
    float*** pool_input_grad = malloc_3d_float_array(pool_layer->input_shape.height, pool_layer->input_shape.width, pool_layer->input_shape.channels);
    float*** fc_input_grad = malloc_3d_float_array(fc_layer->input_shape.height, fc_layer->input_shape.width, fc_layer->input_shape.channels);

    float*** conv_output_grad = malloc_3d_float_array(conv_layer->output_shape.height, conv_layer->output_shape.width, conv_layer->output_shape.channels);
    float*** activation_output_grad = malloc_3d_float_array(activation_layer->output_shape.height, activation_layer->output_shape.width, activation_layer->output_shape.channels);
    float*** pool_output_grad = malloc_3d_float_array(pool_layer->output_shape.height, pool_layer->output_shape.width, pool_layer->output_shape.channels);
    float*** fc_output_grad = malloc_3d_float_array(1, 1, fc_layer->output_shape.channels);

    // Backward pass
    layer_backward_pass(fc_layer, pool_output, fc_output_grad, fc_input_grad);
    layer_backward_pass(pool_layer, activation_output, pool_output_grad, pool_input_grad);
    layer_backward_pass(activation_layer, conv_output, activation_output_grad, activation_input_grad);
    layer_backward_pass(conv_layer, input_tensor, conv_output_grad, conv_input_grad);

    // Clean up
    free_3d_float_array(input_tensor, input_shape.height, input_shape.width);
    free_3d_float_array(conv_output, conv_layer->output_shape.height, conv_layer->output_shape.width);
    free_3d_float_array(activation_output, activation_layer->output_shape.height, activation_layer->output_shape.width);
    free_3d_float_array(pool_output, pool_layer->output_shape.height, pool_layer->output_shape.width);
    free_3d_float_array(fc_output, 1, 1);

    free_3d_float_array(conv_input_grad, input_shape.height, input_shape.width);
    free_3d_float_array(activation_input_grad, activation_layer->input_shape.height, activation_layer->input_shape.width);
    free_3d_float_array(pool_input_grad, pool_layer->input_shape.height, pool_layer->input_shape.width);
    free_3d_float_array(fc_input_grad, 1, 1);

    free_3d_float_array(conv_output_grad, conv_layer->output_shape.height, conv_layer->output_shape.width);
    free_3d_float_array(activation_output_grad, activation_layer->output_shape.height, activation_layer->output_shape.width);
    free_3d_float_array(pool_output_grad, pool_layer->output_shape.height, pool_layer->output_shape.width);
    free_3d_float_array(fc_output_grad, 1, 1);

    delete_layer(conv_layer);
    delete_layer(activation_layer);
    delete_layer(pool_layer);
    delete_layer(fc_layer);

    return 0;
}