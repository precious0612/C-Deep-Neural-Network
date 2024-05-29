//
//  cdnn.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/28/24.
//

#include "cdnn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <json-c/json.h>

#include "model/model.h"

Model* create(int input_width, int input_height, int input_channels,
              int output_width, int output_height, int output_channels) {
    // Check for invalid input dimensions
    if (input_width <= 0  || input_height <= 0  || input_channels <= 0 ||
        output_width <= 0 || output_height <= 0 || output_channels <= 0) {
        fprintf(stderr, "Error: Invalid input or output dimensions.\n");
        return NULL;
    }

    // Setting Input Shape and Output Shape
    Dimensions input_shape  = {input_width, input_height, input_channels};
    Dimensions output_shape = {output_width, output_height, output_channels};

    // Create a new CNN model
    Model* model = create_model(input_shape, output_shape);
    if (model == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for model.\n");
        return NULL;
    }

    return model;
}

static OptimizerType load_optimizer_information(char* type) {
    if (strcmp(type, "SGD") == 0 || strcmp(type, "sgd") == 0) {
        return SGD;
    } else if (strcmp(type, "Adam") == 0 || strcmp(type, "adam") == 0) {
        return ADAM;
    } else if (strcmp(type, "RMSProp") == 0 || strcmp(type, "rmsprop") == 0) {
        return RMSPROP;
    } else {
        fprintf(stderr, "Invalid string for optimizer, please check the model config for optimizer. Default to set as SGD\n");
        return SGD;
    }
}

static char* get_optimizer_information(OptimizerType type) {
    switch (type) {
        case SGD:
            return "sgd";
            
        case ADAM:
            return "adam";
            
        case RMSPROP:
            return "rmsprop";
            
        default:
            return "";
    }
}

static LossType load_loss_information(char* type) {
    if (strcmp(type, "cross entropy") == 0 || strcmp(type, "cross_entropy") == 0 || strcmp(type, "CrossEntropy") == 0 || strcmp(type, "categorical_crossentropy") == 0) {
        return CrossEntropy;
    } else if (strcmp(type, "MSE") == 0 || strcmp(type, "MeanSquareError") == 0 || strcmp(type, "mse") == 0) {
        return MSE;
    } else {
        fprintf(stderr, "Invalid string for loss function, please check the model config for loss. Default to set as MSE\n");
        return MSE;
    }
}

static char* get_loss_information(LossType type) {
    switch (type) {
        case CrossEntropy:
            return "categorical_crossentropy";
        
        case MSE:
            return "mse";
            
        default:
            return "";
    }
}

static Metric load_metric_information(char* type) {
    if (strcmp(type, "loss") == 0) {
        return LOSS;
    } else if (strcmp(type, "accuracy") == 0) {
        return ACCURACY;
    } else {
        fprintf(stderr, "Invalid string for metric, please check the model config for metric. Default to set as Loss\n");
        return LOSS;
    }
}

static char* get_metric_information(Metric metric) {
    switch (metric) {
        case LOSS:
            return "loss";
        
        case ACCURACY:
            return "accuracy";
            
        default:
            return "";
    }
}

static LayerType load_layer_type(const char* type) {
    if (strcmp(type, "convolutional") == 0) {
        return CONVOLUTIONAL;
    } else if (strcmp(type, "pooling") == 0) {
        return POOLING;
    } else if (strcmp(type, "fully_connected") == 0) {
        return FULLY_CONNECTED;
    } else if (strcmp(type, "dropout") == 0) {
        return DROPOUT;
    } else if (strcmp(type, "flatten") == 0) {
        return FLATTEN;
    } else if (strcmp(type, "activation") == 0) {
        return ACTIVATION;
    } else {
        fprintf(stderr, "Invalid string for layer type, please check the model JSON file. Default to set as dropout 0.0\n");
        return DROPOUT;
    }
}

static char* get_layer_type(LayerType type) {
    switch (type) {
        case CONVOLUTIONAL:
            return "convolutional";
        
        case POOLING:
            return "pooling";
            
        case FULLY_CONNECTED:
            return "fully_connected";
            
        case DROPOUT:
            return "dropout";
            
        case FLATTEN:
            return "flatten";
            
        case ACTIVATION:
            return "activation";
            
        default:
            return "";
    }
}

static ActivationType load_activation_information(const char* type) {
    if (strcmp(type, "relu") == 0 || strcmp(type, "ReLU") == 0) {
        return RELU;
    } else if (strcmp(type, "Sigmoid") == 0 || strcmp(type, "sigmoid") == 0) {
        return SIGMOID;
    } else if (strcmp(type, "Tanh") == 0 || strcmp(type, "tanh") == 0) {
        return TANH;
    } else if (strcmp(type, "Softmax") == 0 || strcmp(type, "softmax") == 0) {
        return SOFTMAX;
    } else {
        fprintf(stderr, "Invalid string for activation, please check the layer parameter for activation. Default to set as ReLU\n");
        return RELU;
    }
}

static char* get_activation_information(ActivationType activation) {
    switch (activation) {
        case RELU:
            return "relu";
        
        case SIGMOID:
            return "sigmoid";
            
        case TANH:
            return "tanh";
            
        case SOFTMAX:
            return "softmax";
            
        default:
            return "";
    }
}

static PoolType load_pool_type(const char* type) {
    if (strcmp(type, "max") == 0) {
        return MAX;
    } else if (strcmp(type, "avarage") == 0) {
        return AVARAGE;
    } else {
        fprintf(stderr, "Invalid string for pool type, please check the layer config for pool type. Default to set as avarage\n");
        return AVARAGE;
    }
}

static char* get_pool_type(PoolType type) {
    switch (type) {
        case MAX:
            return "max";
        
        case AVARAGE:
            return "avarage";
            
        default:
            return "";
    }
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
    int input_width    = json_object_get_int(json_object_array_get_idx(input_shape_json, 0));
    int input_height   = json_object_get_int(json_object_array_get_idx(input_shape_json, 1));
    int input_channels = json_object_get_int(json_object_array_get_idx(input_shape_json, 2));

    struct json_object* output_shape_json = json_object_object_get(root, "output_shape");
    int output_width    = 1;
    int output_height   = 1;
    int output_channels = json_object_get_int(json_object_array_get_idx(output_shape_json, 0));

    // Create the model
    Model* model = create(input_width, input_height, input_channels, output_width, output_height, output_channels);
    if (model == NULL) {
        json_object_put(root);
        return NULL;
    }

    // Extract model configuration
    const char* optimizer     = json_object_get_string(json_object_object_get(root, "optimizer"));
    double learning_rate      = json_object_get_double(json_object_object_get(root, "learning_rate"));
    const char* loss_function = json_object_get_string(json_object_object_get(root, "loss_function"));
    const char* metric_name   = json_object_get_string(json_object_object_get(root, "metric_name"));

    ModelConfig config;
    strcpy(config.optimizer, optimizer);
    config.learning_rate = learning_rate;
    strcpy(config.loss_function, loss_function);
    strcpy(config.metric_name, metric_name);

    // Add layers
    struct json_object* layers_json = json_object_object_get(root, "layers");
    int num_layers                  = (int)json_object_array_length(layers_json);
    for (int i = 0; i < num_layers; i++) {
        struct json_object* layer_json  = json_object_array_get_idx(layers_json, i);
        const char* type                = json_object_get_string(json_object_object_get(layer_json, "type"));
        const LayerType layer_type      = load_layer_type(type);
        struct json_object* params_json = NULL;
        int stride                      = 1;
        const char* activation          = NULL;
        
        switch (layer_type) {
            case CONVOLUTIONAL:
                params_json         = json_object_object_get(layer_json, "params");
                int num_filters     = json_object_get_int(json_object_object_get(params_json, "num_filters"));
                int filter_size     = json_object_get_int(json_object_object_get(params_json, "filter_size"));
                stride              = json_object_get_int(json_object_object_get(params_json, "stride"));
                const char* padding = json_object_get_string(json_object_object_get(params_json, "padding"));
                activation          = json_object_get_string(json_object_object_get(params_json, "activation"));
                add_layer(model, layer_type, num_filters, filter_size, stride, (strcmp(padding, "same") == 0) ? 0 : filter_size / 2, load_activation_information(activation), -1, 0.0f);
                break;
                
            case POOLING:
                params_json           = json_object_object_get(layer_json, "params");
                int pool_size         = json_object_get_int(json_object_object_get(params_json, "pool_size"));
                stride                = json_object_get_int(json_object_object_get(params_json, "stride"));
                const char* pool_type = json_object_get_string(json_object_object_get(params_json, "pool_type"));
                add_layer(model, layer_type, 0, pool_size, stride, 0, -1, load_pool_type(pool_type), 0.0f);
                break;
                
            case FULLY_CONNECTED:
                params_json     = json_object_object_get(layer_json, "params");
                int num_neurons = json_object_get_int(json_object_object_get(params_json, "num_neurons"));
                activation      = json_object_get_string(json_object_object_get(params_json, "activation"));
                add_layer(model, layer_type, num_neurons, 0, 0, 0, load_activation_information(activation), -1, 0.0f);
                break;
                
            case DROPOUT:
                params_json        = json_object_object_get(layer_json, "params");
                float dropout_rate = json_object_get_double(json_object_object_get(params_json, "dropout_rate"));
                add_layer(model, layer_type, 0, 0, 0, 0, -1, -1, dropout_rate);
                break;
                
            case ACTIVATION:
                params_json = json_object_object_get(layer_json, "params");
                activation  = json_object_get_string(json_object_object_get(params_json, "activation"));
                add_layer(model, layer_type, 0, 0, 0, 0, load_activation_information(activation), -1, 0.0f);
                break;
                
            case FLATTEN:
                add_layer(model, layer_type, 0, 0, 0, 0, -1, -1, 0.0f);
                break;
                
            default:
                fprintf(stderr, "Error: Invalid layer type specified.\n");
                break;
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
    json_object_object_add(root, "optimizer", json_object_new_string(get_optimizer_information(model->optimizer->type)));
    json_object_object_add(root, "learning_rate", json_object_new_double(model->learning_rate));
    json_object_object_add(root, "loss_function", json_object_new_string(get_loss_information(model->loss->type)));
    json_object_object_add(root, "metric_name", json_object_new_string(get_metric_information(model->metric)));

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
                json_object_object_add(conv_params, "activation", json_object_new_string(get_activation_information(layer->params.conv_params.activation)));
                json_object_object_add(layer_json, "params", conv_params);
                break;
            case POOLING:
                json_object_object_add(layer_json, "type", json_object_new_string("pooling"));
                struct json_object* pool_params = json_object_new_object();
                json_object_object_add(pool_params, "pool_size", json_object_new_int(layer->params.pooling_params.pool_size));
                json_object_object_add(pool_params, "stride", json_object_new_int(layer->params.pooling_params.stride));
                json_object_object_add(pool_params, "pool_type", json_object_new_string(get_pool_type(layer->params.pooling_params.pool_type)));
                json_object_object_add(layer_json, "params", pool_params);
                break;
            case FULLY_CONNECTED:
                json_object_object_add(layer_json, "type", json_object_new_string("fully_connected"));
                struct json_object* fc_params = json_object_new_object();
                json_object_object_add(fc_params, "num_neurons", json_object_new_int(layer->params.fc_params.num_neurons));
                json_object_object_add(fc_params, "activation", json_object_new_string(get_activation_information(layer->params.fc_params.activation)));
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
                json_object_object_add(activation_params, "activation", json_object_new_string(get_activation_information(layer->params.activation_params.activation)));
                json_object_object_add(layer_json, "params", activation_params);
                break;
                
            default:
                json_object_object_add(layer_json, "type", json_object_new_string("none"));
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
    add_layer(model, CONVOLUTIONAL, filters, kernel_size, stride, padding, load_activation_information(activation), -1, 0.0f);
}

void add_pooling_layer(Model* model, int pool_size, int stride, char* pool_type) {
    add_layer(model, POOLING, 0, pool_size, stride, 0, -1, load_pool_type(pool_type), 0.0f);
}

void add_fully_connected_layer(Model* model, int num_neurons, char* activation) {
    add_layer(model, FULLY_CONNECTED, num_neurons, 0, 0, 0, load_activation_information(activation), -1, 0.0f);
}

void add_dropout_layer(Model* model, float dropout_rate) {
    add_layer(model, DROPOUT, 0, 0, 0, 0, -1, -1, dropout_rate);
}

void add_softmax_layer(Model* model) {
    add_layer(model, ACTIVATION, 0, 0, 0, 0, SOFTMAX, -1, 0.0f);
}

void add_relu_layer(Model* model) {
    add_layer(model, ACTIVATION, 0, 0, 0, 0, RELU, -1, 0.0f);
}

void add_sigmoid_layer(Model* model) {
    add_layer(model, ACTIVATION, 0, 0, 0, 0, SIGMOID, -1, 0.0f);
}

void add_tanh_layer(Model* model) {
    add_layer(model, ACTIVATION, 0, 0, 0, 0, TANH, -1, 0.0f);
}

void compile(Model* model, ModelConfig config) {
    compile_model(model, load_optimizer_information(config.optimizer), config.learning_rate, load_loss_information(config.loss_function), load_metric_information(config.metric_name));
}

void save_weights(Model* model, const char* filename) {
    save_model_weights(model, filename);
}

void load_weights(Model* model, const char* filename) {
    load_model_weights(model, filename);
}

Model *load_vgg16(const char *weights_file, int load_pretrained, int num_classes, ModelConfig config) {
    // Create a new model
    Model *model = create(224, 224, 3, 1, 1, num_classes);

    // Add layers to the model
    add_convolutional_layer(model, 64, 3, 1, 1, "relu");
    add_convolutional_layer(model, 64, 3, 1, 1, "relu");
    add_pooling_layer(model, 2, 2, "max");

    add_convolutional_layer(model, 128, 3, 1, 1, "relu");
    add_convolutional_layer(model, 128, 3, 1, 1, "relu");
    add_pooling_layer(model, 2, 2, "max");

    add_convolutional_layer(model, 256, 3, 1, 1, "relu");
    add_convolutional_layer(model, 256, 3, 1, 1, "relu");
    add_convolutional_layer(model, 256, 3, 1, 1, "relu");
    add_pooling_layer(model, 2, 2, "max");

    add_convolutional_layer(model, 512, 3, 1, 1, "relu");
    add_convolutional_layer(model, 512, 3, 1, 1, "relu");
    add_convolutional_layer(model, 512, 3, 1, 1, "relu");
    add_pooling_layer(model, 2, 2, "max");

    add_convolutional_layer(model, 512, 3, 1, 1, "relu");
    add_convolutional_layer(model, 512, 3, 1, 1, "relu");
    add_convolutional_layer(model, 512, 3, 1, 1, "relu");
    add_pooling_layer(model, 2, 2, "max");

    add_flatten_layer(model);
    add_fully_connected_layer(model, 4096, "relu");
    add_dropout_layer(model, 0.5f);
    add_fully_connected_layer(model, 4096, "relu");
    add_dropout_layer(model, 0.5f);
    add_fully_connected_layer(model, num_classes, "softmax");

    // Compile the model
    compile(model, config);

    // Load the weights if required
    if (load_pretrained && weights_file != NULL) {
        load_vgg16_weights(model, weights_file);
    }

    return model;
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

Accuracy evaluate(Model* model, Dataset* dataset) {
    return evaluate_model(model, dataset);
}

Output predict(Model* model, Input input) {
    return forward_pass(model, input);
}

void free_model(Model* model) {
    delete_model(model);
}
