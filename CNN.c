/* CNN.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CNN.h"
#include "model/layer/layer.h"
#include "utils/utils.h"

// Function to create a new CNN model
Model* create(int input_width, int input_height, int input_channels,
              int output_width, int output_height, int output_channels) {
    // Check for invalid input dimensions
    if (input_width <= 0 || input_height <= 0 || input_channels <= 0 ||
        output_width <= 0 || output_height <= 0 || output_channels <= 0) {
        fprintf(stderr, "Error: Invalid input or output dimensions.\n");
        return NULL;
    }

    // Setting Input Shape and Output Shape
    Dimensions input_shape = {input_width, input_height, input_channels};
    Dimensions output_shape = {output_width, output_height, output_channels};

    // Create a new CNN model
    Model* model = create_model(input_shape, output_shape);
    if (model == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for model.\n");
        return NULL;
    }

    return model;
}

void add_convolutional_layer(Model* model, int filters, int kernel_size, int stride, int padding, char* activation) {
    add_layer(model, "convolutional", filters, kernel_size, stride, padding, activation, 0.0f);
}

void add_max_pooling_layer(Model* model, int pool_size, int stride) {
    add_layer(model, "max_pooling", 0, pool_size, stride, 0, NULL, 0.0f);
}

void add_fully_connected_layer(Model* model, int num_neurons, char* activation) {
    add_layer(model, "fully_connected", num_neurons, 0, 0, 0, activation, 0.0f);
}

void add_dropout_layer(Model* model, float dropout_rate) {
    add_layer(model, "dropout", 0, 0, 0, 0, NULL, dropout_rate);
}

void add_flatten_layer(Model* model) {
    add_layer(model, "flatten", 0, 0, 0, 0, NULL, 0.0f);
}

void add_softmax_layer(Model* model) {
    add_layer(model, "activation", 0, 0, 0, 0, "softmax", 0.0f);
}

void add_relu_layer(Model* model) {
    add_layer(model, "activation", 0, 0, 0, 0, "relu", 0.0f);
}

void add_sigmoid_layer(Model* model) {
    add_layer(model, "activation", 0, 0, 0, 0, "sigmoid", 0.0f);
}

void add_tanh_layer(Model* model) {
    add_layer(model, "activation", 0, 0, 0, 0, "tanh", 0.0f);
}

void compile(Model* model, ModelConfig config) {
    compile_model(model, config.optimizer, config.learning_rate, config.loss_function, config.metric_name);
}

void free_model(Model* model) {
    delete_model(model);
}

