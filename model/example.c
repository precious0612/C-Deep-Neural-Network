#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "model.c"
#include "layer/layer.h"
#include "layer/layer.c"
#include "../optimizer/optimizer.h"
#include "../optimizer/optimizer.c"
#include "../dataset.h"
#include "../dataset.c"
#include "../input/data.c"
#include "../utils/tensor.c"
#include "../utils/train.c"
#include "../utils/memory.c"
#include "../utils/loss.c"
#include "../utils/metric.c"
#include "../utils/optim.c"
#include "../utils/tools.c"
#include "../utils/compute/convolution.c"
#include "../utils/compute/pooling.c"
#include "../utils/compute/fully_connected.c"
#include "../utils/compute/dropout.c"
#include "../utils/compute/flatten.c"
#include "../utils/compute/activation.c"
#include "../utils/rand.c"

int main() {
    // Define input and output dimensions
    Dimensions input_dim = {28, 28, 3};
    Dimensions output_dim = {1, 1, 10};

    // Create a new model
    Model* model = create_model(input_dim, output_dim);

    // Add layers to the model
    LayerParams conv_params;
    conv_params.conv_params.num_filters = 32;
    conv_params.conv_params.filter_size = 3;
    conv_params.conv_params.stride = 1;
    conv_params.conv_params.padding = 1;
    strcpy(conv_params.conv_params.activation, "relu");
    Layer* conv_layer = create_layer(CONVOLUTIONAL, conv_params);
    add_layer_to_model(model, conv_layer);

    LayerParams pool_params;
    pool_params.pooling_params.pool_size = 2;
    pool_params.pooling_params.stride = 2;
    strcpy(pool_params.pooling_params.pool_type, "max");
    Layer* pool_layer = create_layer(POOLING, pool_params);
    add_layer_to_model(model, pool_layer);

    LayerParams fc_params;
    fc_params.fc_params.num_neurons = 10;
    strcpy(fc_params.fc_params.activation, "softmax");
    Layer* fc_layer = create_layer(FULLY_CONNECTED, fc_params);
    add_layer_to_model(model, fc_layer);

    // Set optimizer, loss function, and metric
    compile_model(model, "SGD", 0.01f, "mse", "accuracy");

    // Load dataset
    Dataset* dataset = load_dataset_from_json("dataset example/test_data_and_val/dataset.json", input_dim, FLOAT32, 1);
    if (dataset == NULL) {
        fprintf(stderr, "Error: Failed to load dataset\n");
        return 1;
    }

    // Split dataset into batches
    dataset = split_dataset_into_batches(dataset, 2);
    if (dataset == NULL) {
        fprintf(stderr, "Error: Failed to split dataset into batches\n");
        return 1;
    }

    // Train the model
    int num_epochs = 10;
    // // print some data to the console
    // printf("input data: %d %d %d\n", dataset->images[0]->data.float32[0][0][0], dataset->images[3][0][1], dataset->images[27][0][2]);
    train_model(model, dataset, num_epochs);

    // Evaluate the model
    float accuracy = evaluate_model(model, dataset);
    printf("Model accuracy: %.2f%%\n", accuracy * 100);

    // Clean up
    free_dataset(dataset);
    delete_model(model);

    return 0;
}