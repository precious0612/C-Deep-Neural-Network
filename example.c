#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CNN.h"
#include "CNN.c"
#include "model/model.c"
#include "model/layer/layer.c"
#include "optimizer/optimizer.c"
#include "dataset.h"
#include "dataset.c"
#include "input/data.c"
#include "utils/tensor.c"
#include "utils/train.c"
#include "utils/memory.c"
#include "utils/loss.c"
#include "utils/metric.c"
#include "utils/optim.c"
#include "utils/compute/convolution.c"
#include "utils/compute/pooling.c"
#include "utils/compute/fully_connected.c"
#include "utils/compute/dropout.c"
#include "utils/compute/flatten.c"
#include "utils/compute/activation.c"
#include "utils/rand.c"

int main() {
    // Create example model configurations
    ModelConfig config1 = { "SGD", 0.01f, "mse", "accuracy" };
    ModelConfig config2 = { "Adam", 0.001f, "categorical_crossentropy", "accuracy" };

    // Create example input data dimensions
    int input_width = 28, input_height = 28, input_channels = 1;
    int output_width = 1, output_height = 1, output_channels = 10;

    // Create example models
    Model* model1 = create(input_width, input_height, input_channels, output_width, output_height, output_channels);
    Model* model2 = create(input_width, input_height, input_channels, output_width, output_height, output_channels);

    // Add layers to the models
    add_convolutional_layer(model1, 32, 3, 1, 1, "relu");
    add_max_pooling_layer(model1, 2, 2);
    add_fully_connected_layer(model1, 10, "softmax");
    add_convolutional_layer(model2, 32, 3, 1, 1, "relu");
    add_max_pooling_layer(model2, 2, 2);
    add_fully_connected_layer(model2, 5, "softmax");


    // Compile the models
    compile(model1, config1);
    compile(model2, config2);

    // Free memory allocated for the models
    free_model(model1);
    free_model(model2);

    return 0;
}