//
//  dataset.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/16/24.
//

#ifndef dataset_h
#define dataset_h

#include "input/data.h"

#include <stdint.h>

#ifndef __unix__
    char *basename(char *path);
#endif

// TODO: Define the maximum number of images per batch
#define MAX_IMAGES_PER_BATCH 1000
#define MNIST_IMAGE_SIZE     28 * 28
#define MNIST_IMAGE_WIDTH    28
#define MNIST_IMAGE_HEIGHT   28
#define MNIST_IMAGE_CHANNEL  1
#define MNIST_NUM_CLASSES    10
#define PATH_MAX             256

// MARK: - Define Dataset structure

typedef struct Dataset{
    char* name;
    int batch_size;
    int num_images;
    Dimensions data_dimensions;
    DataType data_type;
    InputData** images;
    int* labels;
    struct Dataset* next_batch;
    struct Dataset* val_dataset;
} Dataset;

// MARK: - Methods Declarations

/// Loads a dataset from a JSON file, parsing its content and initializing the dataset structure accordingly.
/// - Parameters:
///   - file_path: A constant character pointer to the JSON file path.
///   - input_dimension: The dimensions of the input data.
///   - data_type: The type of data stored in the dataset (e.g., `SINT32` or `FLOAT32`).
///   - include_val_dataset: An integer flag indicating whether to include a validation dataset (non-zero to include).
/// - Throws: If memory allocation fails at any point or if the JSON content cannot be parsed, appropriate error messages are printed to `stderr`, and `NULL` is returned. All previously allocated memory is freed in case of errors.
/// - Returns: A pointer to the newly created Dataset structure, or NULL if memory allocation fails or the JSON content is invalid.
///
/// - Example Usage:
///     ```c
///     Dataset* dataset = load_dataset_from_json("data/dataset.json", input_dim, FLOAT32, 1);
///     if (dataset == NULL) {
///         // Handle error
///     }
///     ```
///
Dataset* load_dataset_from_json(const char* file_path, Dimensions input_dimension, DataType data_type, int include_val_dataset);

/// Creates a JSON file representing a dataset by scanning a folder for images and optionally including a validation dataset.
/// - Parameters:
///   - folder_path: A constant character pointer to the path of the folder containing the dataset.
///   - include_val_dataset: An integer flag indicating whether to include a validation dataset (non-zero to include).
///   - validation_size: A float specifying the proportion of the dataset to be used as validation, if `include_val_dataset` is zero.
/// - Throws: If any folder path is invalid, the folder cannot be opened, or memory allocation fails, appropriate error messages are printed to `stderr` and the function returns without creating the JSON file.
///
/// - Example Usage:
///     ```c
///     create_dataset_json_file("data/images", 1, 0.2);
///     ```
///
void create_dataset_json_file(const char* folder_path, int include_val_dataset, float validation_size);

/// Allocates and initializes a `Dataset` structure with the specified parameters.
/// - Parameters:
///   - name: A constant character pointer to the name of the dataset. Can be `NULL`.
///   - num_images: The number of images in the dataset.
///   - input_dimensions: The dimensions of the input data.
///   - data_type: The type of data stored in the dataset (e.g., `SINT32` or `FLOAT32`).
/// - Throws: If memory allocation fails at any point, an error message is printed to `stderr`, all previously allocated memory is freed, and `NULL` is returned.
/// - Returns: A pointer to the newly created `Dataset` structure, or `NULL` if memory allocation fails.
///
/// - Example Usage:
///     ```c
///     Dimensions input_dimensions = {640, 480, 3};
///     Dataset* dataset = create_dataset("training_set", 1000, input_dimensions, FLOAT32);
///     if (dataset == NULL) {
///         // Handle error
///     }
///     ```
///
Dataset* create_dataset(const char* name, int num_images, Dimensions input_dimensions, DataType data_type);

/// Splits the original dataset into a specified number of batches and returns the head of a linked list of batch datasets.
/// - Parameters:
///   - original_dataset: A pointer to the original `Dataset` to be split into batches.
///   - num_batches: The number of batches to split the dataset into.
/// - Throws: If memory allocation fails at any point, an error message is printed to `stderr`, all previously allocated memory is freed, and `NULL` is returned. If the number of images in the original dataset is insufficient for the specified number of batches, an error message is printed and `NULL` is returned.
/// - Returns: A pointer to the head of the linked list of batch datasets, or `NULL` if memory allocation fails.
///
/// - Example Usage:
///     ```c
///     Dataset* batch_head = split_dataset_into_batches(original_dataset, 5);
///     if (batch_head == NULL) {
///         // Handle error
///     }
///     ```
///
Dataset* split_dataset_into_batches(Dataset* original_dataset, int num_batches);

/// Prints the information of the given dataset including name, dimensions, data type, and validation dataset information if available.
///
/// - Parameters:
///   - dataset: A pointer to the `Dataset` whose information is to be printed.
///
/// - Example Usage:
///     ```c
///     print_dataset_info(dataset);
///     ```
///
void print_dataset_info(Dataset* dataset);

/// Creates a copy of the given dataset, including all images, labels, and validation datasets, and returns a pointer to the new dataset.
/// - Parameters:
///   - original_dataset: A pointer to the original `Dataset` to be copied.
/// - Throws: If memory allocation fails at any point, an error message is printed to `stderr`, all previously allocated memory is freed, and `NULL` is returned.
/// - Returns: A pointer to the copied `Dataset`, or `NULL` if memory allocation fails.
///
/// - Example Usage:
///     ```c
///     Dataset* copied_dataset = copy_dataset(original_dataset);
///     if (copied_dataset == NULL) {
///         // Handle error
///     }
///     ```
///
Dataset* copy_dataset(Dataset* original_dataset);

/// Frees the memory allocated for the given dataset and all its associated batches, images, labels, and validation datasets.
/// - Parameters:
///   - dataset: A pointer to the `Dataset` to be freed.
///
/// - Example Usage:
///     ```c
///     free_dataset(dataset);
///     ```
///
void free_dataset(Dataset* dataset);

// MARK: - Load MNIST Dataset

/// Creates and returns a pointer to a `Dataset` structure containing the loaded data.
/// - Parameters:
///   - train_images_path: The file path to the training images.
///   - train_labels_path: The file path to the training labels.
///   - test_images_path: The file path to the test images.
///   - test_labels_path: The file path to the test labels.
///   - data_type: The data type of the image data (SINT32 or FLOAT32).
///
/// - Returns:
///   A pointer to the loaded `Dataset` structure if successful, or NULL if an error occurs.
///
/// - Example Usage:
///   ```c
///   Dataset* mnist = load_mnist_dataset(<train_images_path>, <train_labels_path>, <test_images_path>, <test_labels_path>, FLOAT32);
///   ```
///
Dataset* load_mnist_dataset(const char* train_images_path, const char* train_labels_path,
                             const char* test_images_path, const char* test_labels_path,
                             DataType data_type);

#endif /* dataset_h */
