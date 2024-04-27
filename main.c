#include <stdio.h>
#include <stdlib.h>
#include "CNN.h"
#include "dataset.h"

int main() {
    // Load dataset from JSON file
    Dimensions input_dimensions = {28, 28, 3};
    create_dataset_json_file("/Users/precious/Design_Neural_Network/dataset example/test_data_and_val", 1, 0.0f);
    Dataset* dataset = load_dataset_from_json("/Users/precious/Design_Neural_Network/dataset example/test_data_and_val/dataset.json", input_dimensions, FLOAT32, 1);
    if (dataset == NULL) {
        printf("Error loading dataset\n");
        return 1;
    }

    // Split dataset into batches
    dataset = split_dataset_into_batches(dataset, 2);
    if (dataset == NULL) {
        printf("Error splitting dataset into batches\n");
        free_dataset(dataset);
        return 1;
    }

    // Create a new model
    Model* model = create(28, 28, 3, 1, 1, 10);
    if (model == NULL) {
        printf("Error creating model\n");
        free_dataset(dataset);
        return 1;
    }

    // Add layers to the model
    add_convolutional_layer(model, 32, 3, 1, 1, "relu");
    add_max_pooling_layer(model, 2, 2);
    add_convolutional_layer(model, 64, 3, 1, 1, "relu");
    add_max_pooling_layer(model, 2, 2);
    add_flatten_layer(model);
    add_fully_connected_layer(model, 128, "relu");
    add_dropout_layer(model, 0.5f);
    add_fully_connected_layer(model, 10, "softmax");

    // Compile the model
    ModelConfig config = {"Adam", 0.001f, "categorical_crossentropy", "accuracy"};
    compile(model, config);

    // Train the model
    train(model, dataset, 10);

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