/* dataset.h (high-level)

 * This file defines a structure to hold a dataset of images, along with its properties.
 * The dataset is a linked list of images, each with its own label (an integer).
 * This structure is used to represent a dataset of images, and is used throughout the codebase.
 *
 */

#ifndef DATASET_H
#define DATASET_H

#include "input/data.h"

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
    struct Dataset* val_dataset;
} Dataset;

/* 
 * Function to load a dataset from a JSON file.
 *
 * This function reads a JSON file containing dataset information, including image file paths and labels,
 * and loads the dataset into memory. The dataset is represented by a Dataset struct.
 * 
 * Standard .json file format:
 * {
 *   "dataset_name": "test_pic",
 *   "num_images": 2,
 *   "images": [
 *     {
 *       "file_path": "input/test_pic/0/test.jpeg",
 *       "label": "0"
 *     }
 * ,    {
 *       "file_path": "input/test_pic/1/test.png",
 *       "label": "1"
 *     }
 *   ]
 * }
 * You also can use the create_dataset_json_file function to create a JSON file from a folder of images. ( just like `test_pic/dataset.json` )
 *
 * Parameters:
 * - file_path: Path to the JSON file containing dataset information.
 * - input_dimensions: Dimensions of the input data (width, height, channels).
 * - data_type: Type of data in the dataset (Int or FLOAT32).
 * - include_val_dataset: 1 if the dataset contains a val folder include validation dataset, 0 otherwise.
 *
 * Returns:
 * - A pointer to the loaded Dataset struct if successful, NULL otherwise.
 *
 * Note: The returned Dataset struct should be freed using the free_dataset function when no longer needed.
 *
 * Usage example:
 * 
 * Dimensions input_dimensions = {28, 28, 1};
 * Dataset* dataset = load_dataset_from_json("dataset.json", input_dimensions, FLOAT32, 1);
 * if (dataset == NULL) {
 *     // Handle error
 * }
 */
Dataset* load_dataset_from_json(const char* file_path, Dimensions input_dimensions, DataType data_type, int include_val_dataset);

/* 
 * Function to create a JSON file describing a dataset from image files in a folder structure.
 *
 * This function traverses the specified folder structure, counts the number of images,
 * and constructs a JSON file containing information about the dataset, including image file paths and labels.
 * Each subfolder in the specified folder is treated as a separate class, and the folder name is used as the label.
 *
 * Parameters:
 * - folder_path: Path to the root folder containing the dataset images organized in subfolders by class.
 * - include_val_dataset: 1 if the dataset contains a val folder include validation dataset, 0 otherwise.
 * - validation_size: The ratio of the validation dataset (if include_val_dataset is 0, this parameter is ignored).
 * Tips: If you want to create a JSON file for a dataset that contains a val folder, 
 * you set include_val_dataset to 1 and the validation_size will be ignored.
 *
 * Note: The created JSON file will have the following structure:
 * {
 *   "dataset_name": "Name of the dataset",
 *   "num_images": Number of images in the dataset,
 *   "images": [
 *     {
 *       "file_path": "Path to the image file",
 *       "label": "Label of the image (folder name)"
 *     },
 *     ...
 *   ],
 *   "validation_size": Ratio of the validation dataset
 * }
 * Or if include_val_dataset is 1:
 * {
 *   "dataset_name": "Name of the dataset",
 *   "num_images": Number of images in the dataset,
 *   "images": [
 *     {
 *       "file_path": "Path to the image file",
 *       "label": "Label of the image (folder name)"
 *     },
 *     ...
 *   ],
 *   "val_dataset": "Relative path to the validation dataset folder",
 *   "num_val_images": Number of images in the validation dataset,
 *   "val_images": [
 *    {
 *      "file_path": "Path to the image file",
 *     "label": "Label of the image (folder name)"
 *    },
 *    ...
 *   ]
 * }
 *
 * Usage example:
 * 
 * create_dataset_json_file("dataset_folder", 1, 0.0f);
 * create_dataset_json_file("dataset_folder", 0, 0.2f);
 * 
 * P.S. the JSON file will be created in the same folder as the dataset_folder
 */
void create_dataset_json_file(const char* folder_path, int include_val_dataset, float validation_size);

/* 
 * Function to split a dataset into multiple batches in-place.
 *
 * This function divides the given dataset into a specified number of batches,
 * ensuring an approximately equal distribution of images among the batches.
 * It modifies the original dataset to link the batches together.
 *
 * Parameters:
 * - dataset: Pointer to the Dataset struct representing the original dataset.
 * - num_batches: Number of batches to create.
 *
 * Returns:
 * - A pointer to the modified original dataset, now linked with batch datasets, if successful, NULL otherwise.
 *
 * Note: The original dataset is modified in-place to link the batch datasets.
 * The returned pointer points to the head of the linked list of batches.
 * Ensure to free the entire linked list using the free_dataset function when no longer needed.
 *
 * Usage example:
 * 
 * Dataset* dataset = load_dataset_from_json("dataset.json", input_dimensions, FLOAT32);
 * dataset = split_dataset_into_batches(dataset, 5);
 * if (dataset == NULL) {
 *     // Handle error
 * }
 */
Dataset* split_dataset_into_batches(Dataset* dataset, int num_batches);

/* 
 * Function to split a dataset into multiple batches. (No recommend to use this function directly)
 *
 * This function divides the given dataset into a specified number of batches,
 * ensuring an approximately equal distribution of images among the batches.
 * Each batch is represented by a separate Dataset struct.
 *
 * Parameters:
 * - dataset: Pointer to the Dataset struct representing the original dataset.
 * - num_batches: Number of batches to create.
 *
 * Returns:
 * - An array of pointers to the created batch datasets if successful, NULL otherwise.
 *
 * Note: This function is used in Dataset* split_dataset_into_batches(Dataset* dataset, int num_batches);
 * So this function is not recommended to be used directly.
 *
 * Usage example:
 * 
 * Dataset* dataset = load_dataset_from_json("dataset.json", input_dimensions, FLOAT32);
 * Dataset** batches = create_batches(dataset, 5);
 * if (batches == NULL) {
 *     // Handle error
 * }
 */
Dataset** create_batches(const Dataset* dataset, int num_batches);

// Function to free memory allocated for dataset
void free_dataset(Dataset* dataset);

#endif /* DATASET_H */
