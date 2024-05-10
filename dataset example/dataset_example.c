#include <stdio.h>

#include "../dataset.h"
#include "../dataset.c"
#include "../input/data.h"
#include "../input/data.c"

// Define input dimensions
#define INPUT_WIDTH 224
#define INPUT_HEIGHT 224
#define INPUT_CHANNELS 3

int main() {
    // Define input dimensions
    Dimensions input_dimensions = {INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS}; // Example dimensions

    create_dataset_json_file("dataset example/test_data_without_val", 0, 0.5);

    // Load dataset from JSON file
    printf("Loading dataset from JSON file...\n");
    Dataset* dataset = load_dataset_from_json("dataset example/test_data_without_val/dataset.json", input_dimensions, INT, 0);
    if (dataset == NULL) {
        fprintf(stderr, "Error: Failed to load dataset from JSON file\n");
        return 1;
    }
    printf("Dataset loaded successfully\n");

    // Display some information about the loaded dataset
    printf("Dataset Name: %s\n", dataset->name);
    printf("Number of Images: %d\n", dataset->num_images);
    printf("Input Dimensions: %dx%d, Channels: %d\n", dataset->data_dimensions.width, dataset->data_dimensions.height, dataset->data_dimensions.channels);
    printf("Data Type: %s\n", dataset->data_type == INT ? "INT" : "FLOAT32");

    // Display some information about validation dataset
    if (dataset->val_dataset != NULL) {
        printf("Validation Dataset Name: %s\n", dataset->val_dataset->name);
        printf("Number of Images: %d\n", dataset->val_dataset->num_images);
        printf("Input Dimensions: %dx%d, Channels: %d\n", dataset->val_dataset->data_dimensions.width, dataset->val_dataset->data_dimensions.height, dataset->val_dataset->data_dimensions.channels);
        printf("Data Type: %s\n", dataset->val_dataset->data_type == INT ? "INT" : "FLOAT32");
    }

    // Create a new JSON file from dataset
    printf("Creating JSON file from dataset...\n");
    create_dataset_json_file("dataset example/test_data_and_val", 1, 0.5);
    printf("JSON file created successfully\n");

    // Split dataset into batches
    int num_batches = 2; // Example number of batches
    dataset = split_dataset_into_batches(dataset, num_batches);
    if (dataset == NULL) {
        fprintf(stderr, "Error: Failed to split dataset into batches\n");

        return 1;
    }
    printf("Dataset split into %d batches successfully\n", num_batches);

    // Free dataset memory
    free_dataset(dataset);

    return 0;
}