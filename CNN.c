/* CNN.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CNN.h"
#include "model/layer/layer.h"  // Include layer.h to access calculate_output_shape

// Function to create a new CNN model
Model* create_model(int input_width, int input_height, int input_channels,
                    int output_width, int output_height, int output_channels) {
    // Check for invalid input dimensions
    if (input_width <= 0 || input_height <= 0 || input_channels <= 0 ||
        output_width <= 0 || output_height <= 0 || output_channels <= 0) {
        fprintf(stderr, "Error: Invalid input or output dimensions.\n");
        return NULL;
    }

    // Allocate memory for the model
    Model* model = (Model*)malloc(sizeof(Model));
    if (model == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for model.\n");
        return NULL;
    }

    // Initialize other model parameters
    strcpy(model->optimizer, "");
    model->learning_rate = 0.0f;
    strcpy(model->loss_function, "");
    strcpy(model->metric_name, "");
    model->layers = NULL; // No layers added yet

    // Initialize input format information
    model->input.width = input_width;
    model->input.height = input_height;
    model->input.channels = input_channels;

    // Initialize output format information
    model->output.width = output_width;
    model->output.height = output_height;
    model->output.channels = output_channels;

    return model;
}

void add_layer(Model* model, LayerType type, LayerParams params) {
    // Check if the model pointer is valid
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    // Allocate memory for the new layer
    Layer* new_layer = (Layer*)malloc(sizeof(Layer));
    if (new_layer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for layer.\n");
        return;
    }

    // Initialize the layer's fields
    new_layer->type = type;
    new_layer->params = params;
    new_layer->next_layer = NULL;

    // If this is the first layer being added
    if (model->layers == NULL) {
        model->layers = new_layer;
        return;
    }

    // Find the last layer in the model
    Layer* current_layer = model->layers;
    while (current_layer->next_layer != NULL) {
        current_layer = current_layer->next_layer;
    }

    // Append the new layer to the end of the model
    current_layer->next_layer = new_layer;
}

void compile_model(Model* model, ModelConfig config) {
    // Check if the model pointer is valid
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    // Assign the configuration settings to the model
    strcpy(model->optimizer, config.optimizer);
    model->learning_rate = config.learning_rate;
    strcpy(model->loss_function, config.loss_function);
    strcpy(model->metric_name, config.metric_name);

    // Check if the final layer output shape matches the output information
    if (check_output_shape(model)) {
        // Output shape matches
        printf("*********************************************\n");
        printf("\nModel compiled successfully!\n");
        print_model_info(model);
        printf("\n*********************************************\n");
    } else {
        // Output shape does not match
        fprintf(stderr, "Error: Final layer output shape does not match the output information.\n");
    }

}

void print_model_info(Model* model) {
    printf("Model Configuration:\n");
    printf("Input Shape: (%d, %d, %d)\n", model->input.width, model->input.height, model->input.channels);
    printf("Optimizer: %s\n", model->optimizer);
    printf("Learning Rate: %.6f\n", model->learning_rate);
    printf("Loss Function: %s\n", model->loss_function);
    printf("Evaluation Metric: %s\n", model->metric_name);
    
    // Iterate through layers and print their types and parameters
    Dimensions layer_input = model->input;
    Layer* current_layer = model->layers;
    int layer_num = 1;
    while (current_layer != NULL) {
        printf("\nLayer %d: ", layer_num);
        switch (current_layer->type) {
            case CONVOLUTIONAL:
                printf("Convolutional\n");
                printf("  Input Shape: (%d, %d, %d)\n", layer_input.width, layer_input.height, layer_input.channels);
                printf("  Number of Filters: %d\n", current_layer->params.conv_params.num_filters);
                printf("  Filter Size: %d\n", current_layer->params.conv_params.filter_size);
                printf("  Stride: %d\n", current_layer->params.conv_params.stride);
                printf("  Padding: %d\n", current_layer->params.conv_params.padding);
                printf("  Activation Function: %s\n", current_layer->params.conv_params.activation);
                layer_input.width = (layer_input.width - current_layer->params.conv_params.filter_size + 2 * current_layer->params.conv_params.padding) / current_layer->params.conv_params.stride + 1;
                layer_input.height = (layer_input.height - current_layer->params.conv_params.filter_size + 2 * current_layer->params.conv_params.padding) / current_layer-> params.conv_params.stride + 1;
                layer_input.channels = current_layer->params.conv_params.num_filters;
                break;
            case POOLING:
                printf("Pooling\n");
                printf("  Input Shape: (%d, %d, %d)\n", layer_input.width, layer_input.height, layer_input.channels);
                printf("  Pool Size: %d\n", current_layer->params.pooling_params.pool_size);
                printf("  Stride: %d\n", current_layer->params.pooling_params.stride);
                printf("  Pool Type: %s\n", current_layer->params.pooling_params.pool_type);
                layer_input.width = (layer_input.width - current_layer->params.pooling_params.pool_size) / current_layer->params.pooling_params.stride + 1;
                layer_input.height = (layer_input.height - current_layer->params.pooling_params.pool_size) / current_layer->params.pooling_params.stride + 1;
                break;
            case FULLY_CONNECTED:
                printf("Fully Connected\n");
                printf("  Input Shape: (%d, %d, %d)\n", layer_input.width, layer_input.height, layer_input.channels);
                printf("  Number of Neurons: %d\n", current_layer->params.fc_params.num_neurons);
                printf("  Activation Function: %s\n", current_layer->params.fc_params.activation);
                layer_input.width = 1;
                layer_input.height = 1;
                layer_input.channels = current_layer->params.fc_params.num_neurons;
                break;
            case DROPOUT:
                printf("Dropout\n");
                printf("  Input Shape: (%d, %d, %d)\n", layer_input.width, layer_input.height, layer_input.channels);
                printf("  Dropout Rate: %.2f\n", current_layer->params.dropout_params.dropout_rate);
            // Add cases for other layer types as needed
        }
        current_layer = current_layer->next_layer;
        layer_num++;
    }
}

// Function to check if the final layer output shape matches the output information
int check_output_shape(Model *model) {
    // Initialize output shape with input shape from model config
    int output_shape[3];
    output_shape[0] = model->input.width;
    output_shape[1] = model->input.height;
    output_shape[2] = model->input.channels;

    // Iterate through the layers of the model to calculate the output shape
    Layer *current_layer = model->layers;
    do {
        // Based on the layer type, calculate the output shape
        switch (current_layer->type) {
            case CONVOLUTIONAL:
                // Calculate output shape for convolutional layer
                // Output shape = [(input_shape - filter_size + 2 * padding) / stride] + 1
                output_shape[0] = (output_shape[0] - current_layer->params.conv_params.filter_size + 2 * current_layer->params.conv_params.padding) / current_layer->params.conv_params.stride + 1;
                output_shape[1] = (output_shape[1] - current_layer->params.conv_params.filter_size + 2 * current_layer->params.conv_params.padding) / current_layer->params.conv_params.stride + 1;
                output_shape[2] = current_layer->params.conv_params.num_filters;
                break;
            case POOLING:
                // Calculate output shape for pooling layer
                // Output shape = [(input_shape - pool_size) / stride] + 1
                output_shape[0] = (output_shape[0] - current_layer->params.pooling_params.pool_size) / current_layer->params.pooling_params.stride + 1;
                output_shape[1] = (output_shape[1] - current_layer->params.pooling_params.pool_size) / current_layer->params.pooling_params.stride + 1;
                output_shape[2] = output_shape[2];
                break;
            case FULLY_CONNECTED:
                // Calculate output shape for fully connected layer
                output_shape[0] = 1;
                output_shape[1] = 1;
                output_shape[2] = current_layer->params.fc_params.num_neurons;
                break;
            default:
                // Unsupported layer type
                fprintf(stderr, "Error: Unsupported layer type.\n");
                break;
        }
        current_layer = current_layer->next_layer;
    } while (current_layer != NULL);

    // Compare final output shape with output information
    if (output_shape[0] == model->output.width &&
        output_shape[1] == model->output.height &&
        output_shape[2] == model->output.channels) {
        return 1; // Output shape matches
    } else {
        return 0; // Output shape does not match
    }
}

void free_model(Model* model) {
    // Check if the model pointer is valid
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    // Free memory for layers
    Layer* current_layer = model->layers;
    while (current_layer != NULL) {
        Layer* next_layer = current_layer->next_layer;
        free(current_layer);
        current_layer = next_layer;
    }

    // Free memory for the model itself
    free(model);
}

int main() {
    // Create example model configurations
    ModelConfig config1 = { "SGD", 0.01f, "mse", "accuracy" };
    ModelConfig config2 = { "Adam", 0.001f, "categorical_crossentropy", "accuracy" };

    // Create example layer parameters
    LayerParams conv_params = { .conv_params = { 32, 3, 1, 1, "relu" } };
    LayerParams pooling_params = { .pooling_params = { 2, 2, "max" } };
    LayerParams fc1_params = { .fc_params = { 10, "softmax" } };
    LayerParams fc2_params = { .fc_params = { 5, "softmax" } };

    // Create example input data dimensions
    int input_width = 28, input_height = 28, input_channels = 1;
    int output_width = 1, output_height = 1, output_channels = 10;

    // Create example models
    Model* model1 = create_model(input_width, input_height, input_channels, output_width, output_height, output_channels);
    Model* model2 = create_model(input_width, input_height, input_channels, output_width, output_height, output_channels);

    // Add layers to the models
    add_layer(model1, CONVOLUTIONAL, conv_params);
    add_layer(model1, POOLING, pooling_params);
    add_layer(model1, FULLY_CONNECTED, fc1_params);
    add_layer(model2, CONVOLUTIONAL, conv_params);
    add_layer(model2, POOLING, pooling_params);
    add_layer(model2, FULLY_CONNECTED, fc2_params);

    // Compile the models
    compile_model(model1, config1);
    compile_model(model2, config2);

    // Free memory allocated for the models
    free_model(model1);
    free_model(model2);

    return 0;
}

