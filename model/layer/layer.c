/* model/layer/layer.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "layer.h"
#include "../../utils/utils.h"

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
                layer->weights.conv_weights = malloc_4d_float_array(num_filters, filter_size, filter_size, input_channels);
                layer->biases = calloc(num_filters, sizeof(float));

                // Initialize weights with random values (e.g., Xavier initialization)
                float scale = sqrt(2.0 / (float)(filter_size * filter_size * input_channels));
                for (int f = 0; f < num_filters; f++) {
                    for (int h = 0; h < filter_size; h++) {
                        for (int w = 0; w < filter_size; w++) {
                            for (int c = 0; c < input_channels; c++) {
                                layer->weights.conv_weights[f][h][w][c] = scale * rand_uniform(-1.0, 1.0);
                            }
                        }
                    }
                }
            }
            break;

        case FULLY_CONNECTED:
            {
                int num_neurons = layer->params.fc_params.num_neurons;
                int input_size = layer->input_shape.height * layer->input_shape.width * layer->input_shape.channels;

                // Allocate memory for weights and biases
                layer->weights.fc_weights = malloc_2d_float_array(num_neurons, input_size);
                layer->biases = calloc(num_neurons, sizeof(float));

                // Initialize weights with random values (e.g., Xavier initialization)
                float scale = sqrt(2.0 / (float)(input_size));
                for (int n = 0; n < num_neurons; n++) {
                    for (int i = 0; i < input_size; i++) {
                        layer->weights.fc_weights[n][i] = scale * rand_uniform(-1.0, 1.0);
                    }
                }
            }
            break;

        // Add cases for other layer types if needed

        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
}

void layer_forward_pass(Layer* layer, float*** input, float*** output) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            conv_forward(input, output, layer->input_shape, layer->output_shape,
                         layer->params.conv_params.num_filters,
                         layer->params.conv_params.filter_size,
                         layer->params.conv_params.stride,
                         layer->params.conv_params.padding,
                         layer->weights.conv_weights, layer->biases);
            break;
        case POOLING:
            pool_forward(input, output, layer->input_shape, layer->output_shape,
                         layer->params.pooling_params.pool_size,
                         layer->params.pooling_params.stride,
                         layer->params.pooling_params.pool_type);
            break;
        case FULLY_CONNECTED:
            fc_forward(input, output[0], layer->input_shape, layer->output_shape,
                       layer->params.fc_params.num_neurons,
                       layer->weights.fc_weights,
                       layer->biases);
            break;
        case DROPOUT:
            dropout_forward(input, output, layer->input_shape, layer->output_shape,
                            layer->params.dropout_params.dropout_rate);
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
}

void layer_backward_pass(Layer* layer, float*** input, float*** output_grad, float*** input_grad, float learning_rate) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            conv_backward(input, output_grad, input_grad, layer->input_shape, layer->output_shape,
                          layer->params.conv_params.num_filters,
                          layer->params.conv_params.filter_size,
                          layer->params.conv_params.stride,
                          layer->params.conv_params.padding,
                          layer->weights.conv_weights, layer->biases, learning_rate);
            break;
        case POOLING:
            pool_backward(input, output_grad, input_grad, layer->input_shape, layer->output_shape,
                          layer->params.pooling_params.pool_size,
                          layer->params.pooling_params.stride,
                          layer->params.pooling_params.pool_type);
            break;
        case FULLY_CONNECTED:
            if (layer->prev_layer == NULL) {
                fc_backward(input, output_grad[0], input_grad, layer->input_shape, 
                            layer->params.fc_params.num_neurons, layer->weights.fc_weights,
                            layer->biases, learning_rate, 0);
                break;
            }
            fc_backward(input, output_grad[0], input_grad, layer->input_shape, 
                        layer->params.fc_params.num_neurons, layer->weights.fc_weights,
                        layer->biases, learning_rate, layer->prev_layer->type == FULLY_CONNECTED);
            break;
        case DROPOUT:
            dropout_backward(input, output_grad, input_grad, layer->input_shape, layer->output_shape,
                             layer->params.dropout_params.dropout_rate);
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
                int input_channels = layer->input_shape.channels;

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
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
}

void delete_layer(Layer* layer) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            free_4d_float_array(layer->weights.conv_weights, layer->params.conv_params.num_filters, layer->params.conv_params.filter_size, layer->params.conv_params.filter_size, layer->input_shape.channels);
            free(layer->biases);
            break;
        case POOLING:
            break;
        case FULLY_CONNECTED:
            free_2d_float_array(layer->weights.fc_weights, layer->params.fc_params.num_neurons);
            free(layer->biases);
            break;
        case DROPOUT:
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
    free(layer);
}

