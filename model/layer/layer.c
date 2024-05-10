/* model/layer/layer.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "layer.h"

Layer* create_layer(LayerType type, LayerParams params) {
    Layer* new_layer = malloc(sizeof(Layer));
    if (new_layer == NULL) {
        fprintf(stderr, "Error: Could not allocate memory for layer.\n");
        return NULL;
    }

    new_layer->type = type;
    new_layer->params = params;
    new_layer->next_layer = NULL;
    new_layer->prev_layer = NULL;

    return new_layer;
}

void initialize_layer(Layer* layer) {
    // Check if the layer pointer is valid
    if (layer == NULL) {
        fprintf(stderr, "Error: Invalid layer pointer.\n");
        return;
    }

    // Initialize weights and biases based on the layer type
    switch (layer->type) {
        case CONVOLUTIONAL:
            {
                int num_filters = layer->params.conv_params.num_filters;
                int filter_size = layer->params.conv_params.filter_size;
                int input_channels = layer->input_shape.channels;

                // Allocate memory for weights and biases
                layer->weights.conv_weights = malloc_4d_float_array(num_filters, input_channels, filter_size, filter_size);
                layer->biases.biases = calloc(num_filters, sizeof(float));

                // Initialize weights with random values (e.g., Xavier initialization)
                float scale = sqrt(2.0 / (float)(filter_size * filter_size * input_channels));
                for (int f = 0; f < num_filters; f++) {
                    for (int c = 0; c < input_channels; c++) {
                        for (int h = 0; h < filter_size; h++) {
                            for (int w = 0; w < filter_size; w++) {
                                layer->weights.conv_weights[f][c][h][w] = scale * rand_uniform(-1.0, 1.0);
                            }
                        }
                    }
                }

                // Allocate memory for weight gradients and bias gradients
                layer->grads.conv_grads = malloc_4d_float_array(num_filters, input_channels, filter_size, filter_size);
                layer->biases.bias_grads = calloc(num_filters, sizeof(float));
            }
            break;

        case POOLING:
            break;

        case FULLY_CONNECTED:
            {
                int num_neurons = layer->params.fc_params.num_neurons;
                int input_size = layer->input_shape.height * layer->input_shape.width * layer->input_shape.channels;

                // Allocate memory for weights and biases
                layer->weights.fc_weights = malloc_2d_float_array(num_neurons, input_size);
                layer->biases.biases = calloc(num_neurons, sizeof(float));

                // Initialize weights with random values (e.g., Xavier initialization)
                float scale = sqrt(2.0 / (float)(input_size));
                for (int n = 0; n < num_neurons; n++) {
                    for (int i = 0; i < input_size; i++) {
                        layer->weights.fc_weights[n][i] = scale * rand_uniform(-1.0, 1.0);
                    }
                }

                // Allocate memory for weight gradients and bias gradients
                layer->grads.fc_grads = malloc_2d_float_array(num_neurons, input_size);
                layer->biases.bias_grads = calloc(num_neurons, sizeof(float));
            }
            break;
        
        case DROPOUT:
            break;

        case ACTIVATION:
            break;

        case FLATTEN:
            break;

        // Add cases for other layer types if needed

        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
}

float*** layer_forward_pass(Layer* layer, float*** input) {
    float*** output;
    switch (layer->type) {
        case CONVOLUTIONAL:
            output = conv_forward(input, layer->input_shape, layer->output_shape,
                         layer->params.conv_params.num_filters,
                         layer->params.conv_params.filter_size,
                         layer->params.conv_params.stride,
                         layer->params.conv_params.padding,
                         layer->weights.conv_weights, layer->biases.biases);
            if (is_empty_string(layer->params.conv_params.activation)) {
                break;
            }
            // Apply activation function (e.g., ReLU, sigmoid, tanh, etc.)
            output = forward_activation(layer->params.conv_params.activation,
                                      output, layer->output_shape);
            break;
        case POOLING:
            output = pool_forward(input, layer->input_shape, layer->output_shape,
                                  layer->params.pooling_params.pool_size,
                                  layer->params.pooling_params.stride,
                                  layer->params.pooling_params.pool_type);
            break;
        case FULLY_CONNECTED:
            output = (float***)malloc(sizeof(float**) * 1);
            output[0] = (float**)malloc(sizeof(float*) * 1);
            output[0][0] = fc_forward(input, layer->input_shape, layer->params.fc_params.num_neurons,
                       layer->weights.fc_weights, layer->biases.biases);
            if (is_empty_string(layer->params.fc_params.activation)) {
                break;
            }
            // Apply activation function (e.g., ReLU, sigmoid, tanh, etc.)
            output = forward_activation(layer->params.fc_params.activation,
                                      output, layer->output_shape);
            break;
        case DROPOUT:
            output = dropout_forward(input, layer->input_shape, layer->params.dropout_params.dropout_rate);
            break;
        case ACTIVATION:
            output = forward_activation(layer->params.activation_params.activation,
                                      input, layer->input_shape);
            break;
        case FLATTEN:
            output = (float***)malloc(sizeof(float**) * 1);
            output[0] = (float**)malloc(sizeof(float*) * 1);
            output[0][0] = (float*)malloc(sizeof(float) * layer->output_shape.channels * layer->output_shape.height * layer->output_shape.width);
            flatten(input, output[0][0], layer->input_shape);
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            output = copy_3d_array(input, layer->input_shape);
            break;
    }
    return output;
}

void layer_backward_pass(Layer* layer, float*** input, float*** output_grad, float*** input_grad) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            if (not_empty_string(layer->params.conv_params.activation)) {
                backward_activation(layer->params.conv_params.activation,
                                          input, output_grad, output_grad,
                                          layer->output_shape);
            }
            conv_backward(input, output_grad, input_grad, layer->input_shape, layer->output_shape,
                          layer->params.conv_params.num_filters,
                          layer->params.conv_params.filter_size,
                          layer->params.conv_params.stride,
                          layer->params.conv_params.padding,
                          layer->weights.conv_weights, layer->grads.conv_grads, layer->biases.bias_grads);
            break;
        case POOLING:
            pool_backward(input, output_grad, input_grad, layer->input_shape, layer->output_shape,
                          layer->params.pooling_params.pool_size,
                          layer->params.pooling_params.stride,
                          layer->params.pooling_params.pool_type);
            break;
        case FULLY_CONNECTED:
            if (not_empty_string(layer->params.fc_params.activation)) {
                backward_activation(layer->params.fc_params.activation,
                                          input, output_grad, output_grad,
                                          layer->output_shape);
            }
            if (layer->prev_layer == NULL) {
                fc_backward(input, output_grad[0], input_grad, layer->input_shape,
                            layer->params.fc_params.num_neurons, layer->weights.fc_weights,
                            layer->grads.fc_grads, layer->biases.bias_grads, 0);
                break;
            }
            fc_backward(input, output_grad[0], input_grad, layer->input_shape,
                        layer->params.fc_params.num_neurons, layer->weights.fc_weights,
                        layer->grads.fc_grads, layer->biases.bias_grads, 
                        layer->input_shape.height == 1 && layer->input_shape.width == 1);
            break;
        case DROPOUT:
            dropout_backward(input, output_grad, input_grad, layer->input_shape);
            break;
        case ACTIVATION:
            backward_activation(layer->params.activation_params.activation,
                                      input, output_grad, input_grad,
                                      layer->input_shape);
            break;
        case FLATTEN:
            unflatten(output_grad[0][0], input_grad, layer->input_shape);
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
}

void update_layer_weights(Layer* layer, Optimizer* optimizer, int layer_index) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            update_conv_weights(layer->params.conv_params.num_filters, layer->params.conv_params.filter_size,
                                layer->input_shape.channels, layer->weights.conv_weights, 
                                layer->grads.conv_grads, layer->biases.biases, 
                                layer->biases.bias_grads, optimizer, layer_index);
            break;
        case FULLY_CONNECTED:
            update_fc_weights(layer->params.fc_params.num_neurons, layer->weights.fc_weights,
                              layer->grads.fc_grads, layer->biases.biases, layer->biases.bias_grads,
                              optimizer, layer_index);
            break;
        // Add cases for other layer types as needed
        default:
            break;
    }
}

void reset_layer_grads(Layer* layer) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            memset(layer->grads.conv_grads, 0, sizeof(float****) * layer->params.conv_params.num_filters * layer->params.conv_params.filter_size * layer->params.conv_params.filter_size * layer->input_shape.channels);
            memset(layer->biases.bias_grads, 0, sizeof(float) * layer->params.conv_params.num_filters);
            break;
        case POOLING:
            break;
        case FULLY_CONNECTED:
            memset(layer->grads.fc_grads, 0, sizeof(float**) * layer->params.fc_params.num_neurons * layer->output_shape.channels);
            memset(layer->biases.bias_grads, 0, sizeof(float) * layer->params.fc_params.num_neurons);
            break;
        case DROPOUT:
            break;
        case ACTIVATION:
            break;
        case FLATTEN:
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
}

void compute_output_shape(Layer* layer) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            {
                int filter_size = layer->params.conv_params.filter_size;
                int stride = layer->params.conv_params.stride;
                int padding = layer->params.conv_params.padding;
                int num_filters = layer->params.conv_params.num_filters;

                int input_height = layer->input_shape.height;
                int input_width = layer->input_shape.width;

                int output_height = (input_height - filter_size + 2 * padding) / stride + 1;
                int output_width = (input_width - filter_size + 2 * padding) / stride + 1;

                layer->output_shape.height = output_height;
                layer->output_shape.width = output_width;
                layer->output_shape.channels = num_filters;
            }
            break;
        case POOLING:
            {
                int pool_size = layer->params.pooling_params.pool_size;
                int stride = layer->params.pooling_params.stride;

                int input_height = layer->input_shape.height;
                int input_width = layer->input_shape.width;
                int input_channels = layer->input_shape.channels;

                int output_height = (input_height - pool_size) / stride + 1;
                int output_width = (input_width - pool_size) / stride + 1;

                layer->output_shape.height = output_height;
                layer->output_shape.width = output_width;
                layer->output_shape.channels = input_channels;
            }
            break;
        case FULLY_CONNECTED:
            {
                int num_neurons = layer->params.fc_params.num_neurons;

                layer->output_shape.height = 1;
                layer->output_shape.width = 1;
                layer->output_shape.channels = num_neurons;
            }
            break;
        case DROPOUT:
            {
                layer->output_shape = layer->input_shape;
            }
            break;
        case ACTIVATION:
            {
                layer->output_shape = layer->input_shape;
            }
            break;
        case FLATTEN:
            {
                layer->output_shape.height = 1;
                layer->output_shape.width = 1;
                layer->output_shape.channels = layer->input_shape.channels * layer->input_shape.height * layer->input_shape.width;
            }
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
}

void delete_layer(Layer* layer) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            free_4d_float_array(layer->weights.conv_weights, layer->params.conv_params.num_filters, layer->input_shape.channels, layer->params.conv_params.filter_size);
            free_4d_float_array(layer->grads.conv_grads, layer->params.conv_params.num_filters, layer->input_shape.channels, layer->params.conv_params.filter_size);
            free(layer->biases.biases);
            free(layer->biases.bias_grads);
            break;
        case POOLING:
            break;
        case FULLY_CONNECTED:
            free_2d_float_array(layer->weights.fc_weights, layer->params.fc_params.num_neurons);
            free_2d_float_array(layer->grads.fc_grads, layer->params.fc_params.num_neurons);
            free(layer->biases.biases);
            free(layer->biases.bias_grads);
            break;
        case DROPOUT:
            break;
        case ACTIVATION:
            break;
        case FLATTEN:
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
    free(layer);
}
