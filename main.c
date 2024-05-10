#include <stdio.h>
#include <stdlib.h>
#include "CNN.h"
#include "dataset.h"

#include "CNN.c"
#include "model/model.c"
#include "model/layer/layer.c"
#include "optimizer/optimizer.c"
#include "dataset.c"
#include "input/data.c"
#include "utils/tensor.c"
#include "utils/train.c"
#include "utils/memory.c"
#include "utils/loss.c"
#include "utils/metric.c"
#include "utils/optim.c"
#include "utils/tools.c"
#include "utils/compute/convolution.c"
#include "utils/compute/pooling.c"
#include "utils/compute/fully_connected.c"
#include "utils/compute/dropout.c"
#include "utils/compute/flatten.c"
#include "utils/compute/activation.c"
#include "utils/rand.c"

int main() {
    // // Load dataset
    // Dimensions input_dimensions = {28, 28, 1};

    // Load dataset from json file
    // create_dataset_json_file("dataset example/test_data_and_val", 1, 0.0f);
    // Dataset* dataset = load_dataset_from_json("dataset example/test_data_and_val/dataset.json", input_dimensions, FLOAT32, 1);

    // Load pre-trained model
    ModelConfig vgg16_config = {"Adam", 0.0003f, "categorical_crossentropy", "accuracy"};
    Model *vgg16 = load_vgg16("/Users/precious/Design_Neural_Network/VGG16 weights.h5", 1, 1000, vgg16_config);
    if (vgg16 == NULL) {
        printf("Error loading VGG16 model\n");
        return 1;
    }
    save_weights(vgg16, "vgg16_weights.h5");
    free_model(vgg16);

    // Or Load MNIST dataset directly
    const char* train_images_path = "/Users/precious/Design_Neural_Network/dataset example/mnist/train-images-idx3-ubyte.gz";
    const char* train_labels_path = "/Users/precious/Design_Neural_Network/dataset example/mnist/train-labels-idx1-ubyte.gz";
    const char* test_images_path = "/Users/precious/Design_Neural_Network/dataset example/mnist/t10k-images-idx3-ubyte.gz";
    const char* test_labels_path = "/Users/precious/Design_Neural_Network/dataset example/mnist/t10k-labels-idx1-ubyte.gz";

    Dataset* dataset = load_mnist_dataset(train_images_path, train_labels_path,
                                           test_images_path, test_labels_path, FLOAT32);
    if (dataset == NULL) {
        printf("Error loading dataset\n");
        return 1;
    }

    // Split dataset into batches
    dataset = split_dataset_into_batches(dataset, 1875);
    if (dataset == NULL) {
        printf("Error splitting dataset into batches\n");
        free_dataset(dataset);
        return 1;
    }

    // Create a new model
    Model* model = create(28, 28, 1, 1, 1, 10);
    if (model == NULL) {
        printf("Error creating model\n");
        free_dataset(dataset);
        return 1;
    }

    // Add layers to the model
    add_convolutional_layer(model, 3, 3, 1, 1, "relu");
    add_max_pooling_layer(model, 2, 2);
    // add_convolutional_layer(model, 64, 3, 1, 1, "relu");
    // add_max_pooling_layer(model, 2, 2);
    add_flatten_layer(model);
    add_fully_connected_layer(model, 16, "relu");
    // add_dropout_layer(model, 0.5f);
    add_fully_connected_layer(model, 10, "softmax");

    // Compile the model
    ModelConfig config = {"Adam", 0.0003f, "categorical_crossentropy", "accuracy"};
    compile(model, config);

    // Train the model
    train(model, dataset, 3);

    // Evaluate the model
    float accuracy = evaluate(model, dataset->val_dataset);
    printf("FInal Validation Accuracy: %.2f%%\n", accuracy * 100.0f);

    // Save the model
    int result = save_model_to_json(model, "test_model_config.json");
    if (result != 0) {
        printf("Error saving model\n");
    }

    Model* model2 = create_model_from_json("model_config.json");

    // Free memory
    free_model(model);
    free_model(model2);
    free_dataset(dataset);

    return 0;
}