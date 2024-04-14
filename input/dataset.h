#ifndef DATASET_H
#define DATASET_H

#include "input.h"

// Structure to describe dataset
typedef struct Dataset {
    char* name;
    int batch_size;
    int num_images;
    Dimensions data_dimensions;
    DataType data_type;
    InputData** images;
    int* labels;
    struct Dataset* next_batch;
} Dataset;

// Function to load dataset from file (e.g., JSON)
Dataset* load_dataset_from_json(const char* file_path, Dimensions input_dimensions, DataType data_type);

// Function to create a JSON file from dataset
void create_dataset_json_file(const char* folder_path);

// Create batches
Dataset** create_batches(const Dataset* dataset, int num_batches);

// Function to split dataset into batches
Dataset* split_dataset_into_batches(Dataset* dataset, int num_batches);

// Function to free memory allocated for dataset
void free_dataset(Dataset* dataset);

#endif /* DATASET_H */
