#include <stdio.h>

#include "dataset.h"

// Define input dimensions
#define INPUT_WIDTH 224
#define INPUT_HEIGHT 224
#define INPUT_CHANNELS 3

int main() {
    // Define input dimensions
    Dimensions input_dimensions = {INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS}; // Example dimensions

    // Load dataset from JSON file
    printf("Loading dataset from JSON file...\n");
    Dataset* dataset = load_dataset_from_json("/Users/precious/Design_Neural_Network/input/test_pic/dataset.json", input_dimensions, Int);
    if (dataset == NULL) {
        fprintf(stderr, "Error: Failed to load dataset from JSON file\n");
        return 1;
    }
    printf("Dataset loaded successfully\n");

    // Display some information about the loaded dataset
    printf("Dataset Name: %s\n", dataset->name);
    printf("Number of Images: %d\n", dataset->num_images);
    printf("Input Dimensions: %dx%d, Channels: %d\n", dataset->data_dimensions.width, dataset->data_dimensions.height, dataset->data_dimensions.channels);
    printf("Data Type: %s\n", dataset->data_type == Int ? "Int" : "FLOAT32");

    // Create a new JSON file from dataset
    printf("Creating JSON file from dataset...\n");
    create_dataset_json_file("/Users/precious/Design_Neural_Network/input/test_pic");
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