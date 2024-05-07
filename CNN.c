/* CNN.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <json-c/json.h>        // Include cJSON library for parsing JSON

#include "CNN.h"
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

Model* create_model_from_json(const char* filename) {
    // Load the JSON file
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* json_data = (char*)malloc(length + 1);
    fread(json_data, 1, length, file);
    json_data[length] = '\0';
    fclose(file);

    // Parse the JSON data
    struct json_object* root = json_tokener_parse(json_data);
    free(json_data);
    if (root == NULL) {
        fprintf(stderr, "Error: Could not parse JSON data\n");
        return NULL;
    }

    // Extract input and output shapes
    struct json_object* input_shape_json = json_object_object_get(root, "input_shape");
    int input_width = json_object_get_int(json_object_array_get_idx(input_shape_json, 0));
    int input_height = json_object_get_int(json_object_array_get_idx(input_shape_json, 1));
    int input_channels = json_object_get_int(json_object_array_get_idx(input_shape_json, 2));

    struct json_object* output_shape_json = json_object_object_get(root, "output_shape");
    int output_width = 1;
    int output_height = 1;
    int output_channels = json_object_get_int(json_object_array_get_idx(output_shape_json, 0));

    // Create the model
    Model* model = create(input_width, input_height, input_channels, output_width, output_height, output_channels);
    if (model == NULL) {
        json_object_put(root);
        return NULL;
    }

    // Extract model configuration
    const char* optimizer = json_object_get_string(json_object_object_get(root, "optimizer"));
    double learning_rate = json_object_get_double(json_object_object_get(root, "learning_rate"));
    const char* loss_function = json_object_get_string(json_object_object_get(root, "loss_function"));
    const char* metric_name = json_object_get_string(json_object_object_get(root, "metric_name"));

    ModelConfig config;
    strcpy(config.optimizer, optimizer);
    config.learning_rate = learning_rate;
    strcpy(config.loss_function, loss_function);
    strcpy(config.metric_name, metric_name);

    // Add layers
    struct json_object* layers_json = json_object_object_get(root, "layers");
    int num_layers = json_object_array_length(layers_json);
    for (int i = 0; i < num_layers; i++) {
        struct json_object* layer_json = json_object_array_get_idx(layers_json, i);
        const char* type = json_object_get_string(json_object_object_get(layer_json, "type"));

        if (strcmp(type, "convolutional") == 0) {
            struct json_object* params_json = json_object_object_get(layer_json, "params");
            int num_filters = json_object_get_int(json_object_object_get(params_json, "num_filters"));
            int filter_size = json_object_get_int(json_object_object_get(params_json, "filter_size"));
            int stride = json_object_get_int(json_object_object_get(params_json, "stride"));
            const char* padding = json_object_get_string(json_object_object_get(params_json, "padding"));
            const char* activation = json_object_get_string(json_object_object_get(params_json, "activation"));
            add_convolutional_layer(model, num_filters, filter_size, stride, (strcmp(padding, "same") == 0) ? 0 : filter_size / 2, (char*)activation);
        } else if (strcmp(type, "pooling") == 0) {
            struct json_object* params_json = json_object_object_get(layer_json, "params");
            int pool_size = json_object_get_int(json_object_object_get(params_json, "pool_size"));
            int stride = json_object_get_int(json_object_object_get(params_json, "stride"));
            add_max_pooling_layer(model, pool_size, stride);
        } else if (strcmp(type, "fully_connected") == 0) {
            struct json_object* params_json = json_object_object_get(layer_json, "params");
            int num_neurons = json_object_get_int(json_object_object_get(params_json, "num_neurons"));
            const char* activation = json_object_get_string(json_object_object_get(params_json, "activation"));
            add_fully_connected_layer(model, num_neurons, (char*)activation);
        }
    }

    // Compile the model
    compile(model, config);

    json_object_put(root);
    return model;
}

int save_model_to_json(Model* model, const char* filename) {
    // Create the root JSON object
    struct json_object* root = json_object_new_object();
    if (root == NULL) {
        fprintf(stderr, "Error: Could not create JSON object\n");
        return 1;
    }

    // Add input and output shapes
    struct json_object* input_shape = json_object_new_array();
    json_object_array_add(input_shape, json_object_new_int(model->input.width));
    json_object_array_add(input_shape, json_object_new_int(model->input.height));
    json_object_array_add(input_shape, json_object_new_int(model->input.channels));
    json_object_object_add(root, "input_shape", input_shape);

    struct json_object* output_shape = json_object_new_array();
    if (model->output.width != 1) {
        json_object_array_add(output_shape, json_object_new_int(model->output.width));
    }
    if (model->output.height != 1) {
        json_object_array_add(output_shape, json_object_new_int(model->output.height));
    }
    json_object_array_add(output_shape, json_object_new_int(model->output.channels));
    json_object_object_add(root, "output_shape", output_shape);

    // Add optimizer, learning rate, loss function, and metric
    json_object_object_add(root, "optimizer", json_object_new_string(model->optimizer_name));
    json_object_object_add(root, "learning_rate", json_object_new_double(model->learning_rate));
    json_object_object_add(root, "loss_function", json_object_new_string(model->loss_function));
    json_object_object_add(root, "metric_name", json_object_new_string(model->metric_name));

    // Add layers
    struct json_object* layers = json_object_new_array();
    for (int i = 0; i < model->num_layers; i++) {
        Layer* layer = model->layers[i];
        struct json_object* layer_json = json_object_new_object();

        switch (layer->type) {
            case CONVOLUTIONAL:
                json_object_object_add(layer_json, "type", json_object_new_string("convolutional"));
                struct json_object* conv_params = json_object_new_object();
                json_object_object_add(conv_params, "num_filters", json_object_new_int(layer->params.conv_params.num_filters));
                json_object_object_add(conv_params, "filter_size", json_object_new_int(layer->params.conv_params.filter_size));
                json_object_object_add(conv_params, "stride", json_object_new_int(layer->params.conv_params.stride));
                json_object_object_add(conv_params, "padding", json_object_new_string((layer->params.conv_params.padding == 0) ? "same" : "valid"));
                json_object_object_add(conv_params, "activation", json_object_new_string(layer->params.conv_params.activation));
                json_object_object_add(layer_json, "params", conv_params);
                break;
            case POOLING:
                json_object_object_add(layer_json, "type", json_object_new_string("pooling"));
                struct json_object* pool_params = json_object_new_object();
                json_object_object_add(pool_params, "pool_size", json_object_new_int(layer->params.pooling_params.pool_size));
                json_object_object_add(pool_params, "stride", json_object_new_int(layer->params.pooling_params.stride));
                json_object_object_add(pool_params, "pool_type", json_object_new_string(layer->params.pooling_params.pool_type));
                json_object_object_add(layer_json, "params", pool_params);
                break;
            case FULLY_CONNECTED:
                json_object_object_add(layer_json, "type", json_object_new_string("fully_connected"));
                struct json_object* fc_params = json_object_new_object();
                json_object_object_add(fc_params, "num_neurons", json_object_new_int(layer->params.fc_params.num_neurons));
                json_object_object_add(fc_params, "activation", json_object_new_string(layer->params.fc_params.activation));
                json_object_object_add(layer_json, "params", fc_params);
                break;
            case DROPOUT:
                json_object_object_add(layer_json, "type", json_object_new_string("dropout"));
                struct json_object* dropout_params = json_object_new_object();
                json_object_object_add(dropout_params, "dropout_rate", json_object_new_double(layer->params.dropout_params.dropout_rate));
                json_object_object_add(layer_json, "params", dropout_params);
                break;
            case FLATTEN:
                json_object_object_add(layer_json, "type", json_object_new_string("flatten"));
                break;
            case ACTIVATION:
                json_object_object_add(layer_json, "type", json_object_new_string("activation"));
                struct json_object* activation_params = json_object_new_object();
                json_object_object_add(activation_params, "activation", json_object_new_string(layer->params.activation_params.activation));
                json_object_object_add(layer_json, "params", activation_params);
                break;
        }

        json_object_array_add(layers, layer_json);
    }
    json_object_object_add(root, "layers", layers);

    // Save the JSON data to a file
    const char* json_data = json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY);
    if (json_data == NULL) {
        fprintf(stderr, "Error: Could not print JSON data\n");
        json_object_put(root);
        return 1;
    }

    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        json_object_put(root);
        return 1;
    }

    fputs(json_data, file);
    fclose(file);
    json_object_put(root);

    return 0;
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

void train(Model* model, Dataset* dataset, int epochs) {
    char continue_training;
    printf("Do you want to continue training? (y/n): ");
    scanf(" %c", &continue_training);

    if (continue_training == 'y' || continue_training == 'Y') {
        train_model(model, dataset, epochs);
    } else {
        printf("Training terminated.\n");
    }
}

float evaluate(Model* model, Dataset* dataset) {
    return evaluate_model(model, dataset);
}

float*** predict(Model* model, float*** input) {
    return forward_pass(model, input);
}

void free_model(Model* model) {
    delete_model(model);
}

