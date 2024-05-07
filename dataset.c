/* dataset.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <stdint.h>
#include <zlib.h>
#include "dataset.h"

#include <json-c/json.h>

// Define the maximum number of images per batch
#define MAX_IMAGES_PER_BATCH 100
#define MNIST_IMAGE_SIZE 28 * 28
#define MNIST_NUM_CLASSES 10

Dataset* load_dataset_from_json(const char* file_path, Dimensions input_dimensions, DataType data_type, int include_val_dataset) {
    // Open the JSON file for reading
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_path);
        return NULL;
    }

    // Read the JSON file content
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* json_buffer = (char*)malloc(file_size + 1);
    fread(json_buffer, 1, file_size, file);
    fclose(file);
    json_buffer[file_size] = '\0';

    // Parse JSON
    json_object* root = json_tokener_parse(json_buffer);
    free(json_buffer);
    if (root == NULL) {
        fprintf(stderr, "Error: Unable to parse JSON\n");
        return NULL;
    }

    // Allocate memory for Dataset struct
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (dataset == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        json_object_put(root);
        return NULL;
    }

    // Initialize dataset fields
    dataset->name = NULL;
    dataset->batch_size = 0;
    dataset->num_images = 0;
    dataset->data_dimensions = input_dimensions;
    dataset->data_type = data_type;
    dataset->images = NULL;
    dataset->labels = NULL;
    dataset->next_batch = NULL;
    dataset->val_dataset = NULL;

    // Read dataset information from JSON object
    json_object* dataset_name_obj = json_object_object_get(root, "dataset_name");
    if (dataset_name_obj != NULL) {
        const char* dataset_name = json_object_get_string(dataset_name_obj);
        dataset->name = strdup(dataset_name);
    }

    json_object* num_images_obj = json_object_object_get(root, "num_images");
    if (num_images_obj != NULL) {
        dataset->num_images = json_object_get_int(num_images_obj);
        dataset->batch_size = dataset->num_images;
        // Allocate memory for images and labels
        dataset->images = (InputData**)malloc(dataset->num_images * sizeof(InputData*));
        dataset->labels = (int*)malloc(dataset->num_images * sizeof(int));

        // Read images array
        json_object* images_array_obj = json_object_object_get(root, "images");
        if (images_array_obj != NULL) {
            int index = 0;
            int array_len = json_object_array_length(images_array_obj);
            for (index = 0; index < array_len; ++index) {
                json_object* image_obj = json_object_array_get_idx(images_array_obj, index);
                if (image_obj != NULL) {
                    json_object* file_path_obj = json_object_object_get(image_obj, "file_path");
                    json_object* label_obj = json_object_object_get(image_obj, "label");
                    if (file_path_obj != NULL && label_obj != NULL) {
                        const char* file_path = json_object_get_string(file_path_obj);
                        int label = json_object_get_int(label_obj);
                        // Load image data
                        dataset->images[index] = load_input_data_from_image(file_path, &input_dimensions, data_type);
                        dataset->labels[index] = label;
                    }
                }
            }
        }
    }

    // Read validation dataset information
    if (include_val_dataset) {
        json_object* val_dataset_obj = json_object_object_get(root, "val_dataset");
        if (val_dataset_obj != NULL) {
            // const char* val_dataset_name = json_object_get_string(val_dataset_obj);
            json_object* num_val_images_obj = json_object_object_get(root, "num_val_images");
            if (num_val_images_obj != NULL) {
                int num_val_images = json_object_get_int(num_val_images_obj);
                dataset->val_dataset = (Dataset*)malloc(sizeof(Dataset));
                dataset->val_dataset->name = "val";
                dataset->val_dataset->batch_size = num_val_images;
                dataset->val_dataset->num_images = num_val_images;
                dataset->val_dataset->data_dimensions = input_dimensions;
                dataset->val_dataset->data_type = data_type;
                dataset->val_dataset->images = (InputData**)malloc(num_val_images * sizeof(InputData*));
                dataset->val_dataset->labels = (int*)malloc(num_val_images * sizeof(int));
                dataset->val_dataset->next_batch = NULL;
                dataset->val_dataset->val_dataset = NULL;

                // Read validation images array
                json_object* val_images_array_obj = json_object_object_get(root, "val_images");
                if (val_images_array_obj != NULL) {
                    int index = 0;
                    int array_len = json_object_array_length(val_images_array_obj);
                    for (index = 0; index < array_len; ++index) {
                        json_object* image_obj = json_object_array_get_idx(val_images_array_obj, index);
                        if (image_obj != NULL) {
                            json_object* file_path_obj = json_object_object_get(image_obj, "file_path");
                            json_object* label_obj = json_object_object_get(image_obj, "label");
                            if (file_path_obj != NULL && label_obj != NULL) {
                                const char* file_path = json_object_get_string(file_path_obj);
                                int label = json_object_get_int(label_obj);
                                // Load image data
                                dataset->val_dataset->images[index] = load_input_data_from_image(file_path, &input_dimensions, data_type);
                                dataset->val_dataset->labels[index] = label;
                            }
                        }
                    }
                }
            }
        }
    } else {
        json_object* validation_size_obj = json_object_object_get(root, "validation_size");
        if (validation_size_obj != NULL) {
            double validation_size = json_object_get_double(validation_size_obj);
            int num_val_images = (int)(dataset->num_images * validation_size);
            int num_train_images = dataset->num_images - num_val_images;

            // Split the dataset into training and validation sets
            Dataset* train_dataset = (Dataset*)malloc(sizeof(Dataset));
            train_dataset->name = strdup(dataset->name);
            train_dataset->batch_size = num_train_images;
            train_dataset->num_images = num_train_images;
            train_dataset->data_dimensions = input_dimensions;
            train_dataset->data_type = data_type;
            train_dataset->images = (InputData**)malloc(num_train_images * sizeof(InputData*));
            train_dataset->labels = (int*)malloc(num_train_images * sizeof(int));
            train_dataset->next_batch = NULL;

            dataset->val_dataset = (Dataset*)malloc(sizeof(Dataset));
            dataset->val_dataset->name = "val";
            dataset->val_dataset->batch_size = num_val_images;
            dataset->val_dataset->num_images = num_val_images;
            dataset->val_dataset->data_dimensions = input_dimensions;
            dataset->val_dataset->data_type = data_type;
            dataset->val_dataset->images = (InputData**)malloc(num_val_images * sizeof(InputData*));
            dataset->val_dataset->labels = (int*)malloc(num_val_images * sizeof(int));
            dataset->val_dataset->next_batch = NULL;
            dataset->val_dataset->val_dataset = NULL;

            int train_index = 0;
            int val_index = 0;
            for (int i = 0; i < dataset->num_images; i++) {
                if (i < num_train_images) {
                    train_dataset->images[train_index] = dataset->images[i];
                    train_dataset->labels[train_index] = dataset->labels[i];
                    train_index++;
                } else {
                    dataset->val_dataset->images[val_index] = dataset->images[i];
                    dataset->val_dataset->labels[val_index] = dataset->labels[i];
                    val_index++;
                }
            }

            free(dataset->images);
            free(dataset->labels);
            dataset->images = train_dataset->images;
            dataset->labels = train_dataset->labels;
            dataset->batch_size = num_train_images;
            dataset->num_images = num_train_images;
            dataset->next_batch = NULL;
            free(train_dataset);
        }
    }

    // Cleanup
    json_object_put(root);

    return dataset;
}

void create_dataset_json_file(const char* folder_path, int include_val_dataset, float validation_size) {
    // Open the folder
    DIR* dir = opendir(folder_path);
    if (dir == NULL) {
        fprintf(stderr, "Error: Unable to open folder\n");
        return;
    }

    // Count the number of images and validation images
    int num_images = 0;
    int num_val_images = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            char subfolder_path[256];
            snprintf(subfolder_path, sizeof(subfolder_path), "%s/%s", folder_path, entry->d_name);
            DIR* subfolder = opendir(subfolder_path);
            if (subfolder != NULL) {
                struct dirent* subentry;
                while ((subentry = readdir(subfolder)) != NULL) {
                    if (subentry->d_type == DT_REG) {
                        if (include_val_dataset && strcmp(entry->d_name, "val") == 0) {
                            num_val_images++;
                        } else {
                            num_images++;
                        }
                    } else if (subentry->d_type == DT_DIR && strcmp(entry->d_name, "val") == 0) {
                        char sub_subfolder_path[256];
                        snprintf(sub_subfolder_path, sizeof(sub_subfolder_path), "%s/%s", subfolder_path, subentry->d_name);
                        DIR* sub_subfolder = opendir(sub_subfolder_path);
                        if (sub_subfolder != NULL) {
                            struct dirent* sub_subentry;
                            while ((sub_subentry = readdir(sub_subfolder)) != NULL) {
                                if (sub_subentry->d_type == DT_REG) {
                                    num_val_images++;
                                }
                            }
                            closedir(sub_subfolder);
                        }
                    }
                }
                closedir(subfolder);
            }
        }
    }

    // Construct the JSON file path
    char json_file_path[256];
    snprintf(json_file_path, sizeof(json_file_path), "%s/dataset.json", folder_path);

    // Open the JSON file for writing
    FILE* file = fopen(json_file_path, "w");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to create JSON file\n");
        closedir(dir);
        return;
    }

    // Write dataset information to JSON file
    fprintf(file, "{\n");
    fprintf(file, "  \"dataset_name\": \"%s\",\n", strrchr(folder_path, '/') + 1);
    fprintf(file, "  \"num_images\": %d,\n", num_images);
    fprintf(file, "  \"images\": [\n");

    // Reset the directory stream pointer to the beginning
    rewinddir(dir);

    // Write image information to JSON file
    bool first_image = true;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 && strcmp(entry->d_name, "val") != 0) {
            char subfolder_path[256];
            snprintf(subfolder_path, sizeof(subfolder_path), "%s/%s", folder_path, entry->d_name);
            DIR* subfolder = opendir(subfolder_path);
            if (subfolder != NULL) {
                struct dirent* subentry;
                while ((subentry = readdir(subfolder)) != NULL) {
                    if (subentry->d_type == DT_REG) {
                        // Write file information to JSON file
                        fprintf(file, "%s    {\n", first_image ? "" : ",");
                        fprintf(file, "      \"file_path\": \"%s/%s\",\n", subfolder_path, subentry->d_name);
                        fprintf(file, "      \"label\": \"%s\"\n", entry->d_name); // Assuming folder names are labels
                        fprintf(file, "    }\n");
                        first_image = false;
                    }
                }
                closedir(subfolder);
            }
        }
    }

    // Close the JSON array
    fprintf(file, "  ]\n");

    // Write validation dataset information if include_val_dataset is true
    if (include_val_dataset && num_val_images > 0) {
        fprintf(file, ",\n  \"val_dataset\": \"%s\",\n", "val");
        fprintf(file, "  \"num_val_images\": %d,\n", num_val_images - 1);
        fprintf(file, "  \"val_images\": [\n");

        // Reset the directory stream pointer to the beginning
        rewinddir(dir);

        bool first_val_image = true;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 && strcmp(entry->d_name, "val") == 0) {
                char subfolder_path[256];
                snprintf(subfolder_path, sizeof(subfolder_path), "%s/%s", folder_path, entry->d_name);
                DIR* subfolder = opendir(subfolder_path);
                if (subfolder != NULL) {
                    struct dirent* subentry;
                    while ((subentry = readdir(subfolder)) != NULL) {
                        if (subentry->d_type == DT_DIR && strcmp(subentry->d_name, "..") != 0) {
                            char sub_subfolder_path[256];
                            snprintf(sub_subfolder_path, sizeof(sub_subfolder_path), "%s/%s", subfolder_path, subentry->d_name);
                            DIR* sub_subfolder = opendir(sub_subfolder_path);
                            if (sub_subfolder != NULL) {
                                struct dirent* sub_subentry;
                                while ((sub_subentry = readdir(sub_subfolder)) != NULL) {
                                    if (sub_subentry->d_type == DT_REG) {
                                        // Write file information to JSON file
                                        fprintf(file, "%s    {\n", first_val_image ? "" : ",");
                                        fprintf(file, "      \"file_path\": \"%s/%s\",\n", sub_subfolder_path, sub_subentry->d_name);
                                        fprintf(file, "      \"label\": \"%s\"\n", subentry->d_name); // Assuming subfolder names are labels
                                        fprintf(file, "    }\n");
                                        first_val_image = false;
                                    }
                                }
                                closedir(sub_subfolder);
                            }
                        }
                    }
                    closedir(subfolder);
                }
            }
        }

        // Close the JSON array
        fprintf(file, "  ]\n");
    } else {
        if (validation_size > 0) {
            fprintf(file, ",\n  \"validation_size\": %.2f\n", validation_size);
        } else {
            fprintf(file, ",\n  \"validation_size\": 0.2 \n"); // Default validation size if not provided
        }
    }

    // Close the JSON object
    fprintf(file, "}\n");

    // Close the file and folder
    fclose(file);
    closedir(dir);
}

// Function to split dataset into batches
Dataset** create_batches(const Dataset* dataset, int num_batches) {
    // Calculate the number of images per batch
    int images_per_batch = dataset->num_images / num_batches;
    if (images_per_batch == 0) {
        fprintf(stderr, "Error: Insufficient number of images for the specified number of batches\n");
        return NULL;
    }

    // Allocate memory for the array of batch datasets
    Dataset** batches = (Dataset**)malloc(num_batches * sizeof(Dataset*));
    if (batches == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    // Split the dataset into batches
    int remaining_images = dataset->num_images;
    for (int i = 0; i < num_batches; i++) {
        // Calculate the number of images in this batch
        int batch_size = (i == num_batches - 1) ? remaining_images : images_per_batch;

        // Allocate memory for the batch dataset
        batches[i] = (Dataset*)malloc(sizeof(Dataset));
        if (batches[i] == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            // Free previously allocated batches
            for (int j = 0; j < i; j++) {
                free_dataset(batches[j]);
            }
            free(batches);
            return NULL;
        }

        // Copy dataset information to the batch dataset
        batches[i]->name = dataset->name;
        batches[i]->batch_size = batch_size;
        batches[i]->num_images = batch_size;
        batches[i]->data_dimensions = dataset->data_dimensions;
        batches[i]->data_type = dataset->data_type;
        batches[i]->next_batch = NULL;

        // Allocate memory for images and labels
        batches[i]->images = (InputData**)malloc(batch_size * sizeof(InputData*));
        batches[i]->labels = (int*)malloc(batch_size * sizeof(int));
        if (batches[i]->images == NULL || batches[i]->labels == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            // Free previously allocated batches
            for (int j = 0; j <= i; j++) {
                free_dataset(batches[j]);
            }
            free(batches);
            return NULL;
        }

        // Copy images and labels to the batch dataset
        for (int j = 0; j < batch_size; j++) {
            batches[i]->images[j] = dataset->images[i * images_per_batch + j];
            batches[i]->labels[j] = dataset->labels[i * images_per_batch + j];
        }

        // Update remaining images
        remaining_images -= batch_size;
    }
    
    return batches;
}

Dataset* split_dataset_into_batches(Dataset* original_dataset, int num_batches) {
    Dataset** batches = create_batches(original_dataset, num_batches);
    if (batches == NULL) {
        return NULL;
    }

    // Create a new dataset to hold the batches
    Dataset* new_dataset = (Dataset*)malloc(sizeof(Dataset));
    if (new_dataset == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        for (int i = 0; i < num_batches; i++) {
            free_dataset(batches[i]);
        }
        free(batches);
        return NULL;
    }

    // Copy original dataset information to the new dataset
    new_dataset = batches[0];

    // Link the batches together
    for (int i = 0; i < num_batches - 1; i++) {
        batches[i]->next_batch = batches[i + 1];
    }
    batches[num_batches - 1]->next_batch = NULL;

    free(batches);

    return new_dataset;
}

// Function to free memory allocated for dataset
void free_dataset(Dataset* dataset) {
    if (dataset == NULL) {
        return;
    }

    while (dataset->next_batch != NULL) {

        Dataset* next = dataset->next_batch;
        if (dataset->images != NULL) {
        for (int i = 0; i < dataset->num_images; i++) {
            free_image_data(dataset->images[i], dataset->data_dimensions, dataset->data_type);
        }
        free(dataset->images);
        }
        if (dataset->labels != NULL) {
            free(dataset->labels);
        }

        // Free dataset name
        if (dataset->name != NULL) {
            free(dataset->name);
        }
        free(dataset);
        dataset = next;
    
    }

    if (dataset->val_dataset != NULL) {
        free_dataset(dataset->val_dataset);
    }

}

Dataset* load_mnist_dataset(const char* train_images_path, const char* train_labels_path,
                             const char* test_images_path, const char* test_labels_path,
                             DataType data_type) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (dataset == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    dataset->name = "MNIST";
    dataset->data_dimensions = (Dimensions){28, 28, 1};
    dataset->data_type = data_type;
    dataset->next_batch = NULL;
    dataset->val_dataset = NULL;

    // Load training data
    int num_train_images;
    void** train_images = NULL;
    if (data_type == Int) {
        train_images = (void**)load_mnist_images_int(train_images_path, &num_train_images);
    } else {
        train_images = (void**)load_mnist_images_float(train_images_path, &num_train_images);
    }
    if (train_images == NULL) {
        free_dataset(dataset);
        return NULL;
    }

    int num_train_labels;
    uint8_t* train_labels = load_mnist_labels(train_labels_path, &num_train_labels);
    if (train_labels == NULL) {
        free_mnist_images(train_images, num_train_images, data_type);
        free_dataset(dataset);
        return NULL;
    }

    dataset->batch_size = num_train_images;
    dataset->num_images = num_train_images;
    dataset->images = (InputData**)malloc(num_train_images * sizeof(InputData*));
    dataset->labels = (int*)malloc(num_train_images * sizeof(int));

    printf("Loading training data: ");
    for (int i = 0; i < num_train_images; i++) {
        dataset->images[i] = (InputData*)malloc(sizeof(InputData));
        if (data_type == Int) {
            dataset->images[i]->int_data = (int***)malloc(28 * sizeof(int**));
            for (int j = 0; j < 28; j++) {
                dataset->images[i]->int_data[j] = (int**)malloc(28 * sizeof(int*));
                for (int k = 0; k < 28; k++) {
                    dataset->images[i]->int_data[j][k] = (int*)malloc(sizeof(int));
                    dataset->images[i]->int_data[j][k][0] = ((int*)train_images[i])[j * 28 + k];
                }
            } 
        } else if (data_type == FLOAT32) {
            dataset->images[i]->float32_data = (float***)malloc(28 * sizeof(float**));
            for (int j = 0; j < 28; j++) {
                dataset->images[i]->float32_data[j] = (float**)malloc(28 * sizeof(float*));
                for (int k = 0; k < 28; k++) {
                    dataset->images[i]->float32_data[j][k] = (float*)malloc(sizeof(float));
                    dataset->images[i]->float32_data[j][k][0] = ((float*)train_images[i])[j * 28 + k];
                }
            }
        }
        dataset->labels[i] = (int)train_labels[i]; // Cast to int
        print_progress_bar((float)(i + 1) / num_train_images);
    }
    printf("\n");

    free(train_labels);

    // Load test data
    int num_test_images;
    void** test_images = NULL;
    if (data_type == Int) {
        test_images = (void**)load_mnist_images_int(test_images_path, &num_test_images);
    } else {
        test_images = (void**)load_mnist_images_float(test_images_path, &num_test_images);
    }
    if (test_images == NULL) {
        free_dataset(dataset);
        return NULL;
    }

    int num_test_labels;
    uint8_t* test_labels = load_mnist_labels(test_labels_path, &num_test_labels);
    if (test_labels == NULL) {
        free_mnist_images(test_images, num_test_images, data_type);
        free_dataset(dataset);
        return NULL;
    }

    dataset->val_dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->val_dataset->name = "MNIST Test";
    dataset->val_dataset->data_dimensions = dataset->data_dimensions;
    dataset->val_dataset->data_type = dataset->data_type;
    dataset->val_dataset->batch_size = num_test_images;
    dataset->val_dataset->num_images = num_test_images;
    dataset->val_dataset->images = (InputData**)malloc(num_test_images * sizeof(InputData*));
    dataset->val_dataset->labels = (int*)malloc(num_test_images * sizeof(int));
    dataset->val_dataset->next_batch = NULL;
    dataset->val_dataset->val_dataset = NULL;

    printf("Loading test data: ");
    for (int i = 0; i < num_test_images; i++) {
        dataset->val_dataset->images[i] = (InputData*)malloc(sizeof(InputData));
        if (data_type == Int) {
            dataset->val_dataset->images[i]->int_data = (int***)malloc(28 * sizeof(int**));
            for (int j = 0; j < 28; j++) {
                dataset->val_dataset->images[i]->int_data[j] = (int**)malloc(28 * sizeof(int*));
                for (int k = 0; k < 28; k++) {
                    dataset->val_dataset->images[i]->int_data[j][k] = (int*)malloc(sizeof(int));
                    dataset->val_dataset->images[i]->int_data[j][k][0] = ((int*)test_images[i])[j * 28 + k];
                }
            } 
        } else if (data_type == FLOAT32) {
            dataset->val_dataset->images[i]->float32_data = (float***)malloc(28 * sizeof(float**));
            for (int j = 0; j < 28; j++) {
                dataset->val_dataset->images[i]->float32_data[j] = (float**)malloc(28 * sizeof(float*));
                for (int k = 0; k < 28; k++) {
                    dataset->val_dataset->images[i]->float32_data[j][k] = (float*)malloc(sizeof(float));
                    dataset->val_dataset->images[i]->float32_data[j][k][0] = ((float*)test_images[i])[j * 28 + k];
                }
            }
        }
        dataset->val_dataset->labels[i] = (int)test_labels[i]; // Cast to int
        print_progress_bar((float)(i + 1) / num_test_images);
    }
    printf("\n");

    printf("Finished loading MNIST Datset!\n");

    free(test_labels);

    return dataset;
}

int** load_mnist_images_int(const char* file_path, int* num_images) {
    gzFile file = gzopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_path);
        return NULL;
    }

    uint32_t magic_number;
    if (gzread(file, &magic_number, sizeof(uint32_t)) != sizeof(uint32_t)) {
        fprintf(stderr, "Error: Failed to read magic number\n");
        gzclose(file);
        return NULL;
    }

    // Handle endianness
    magic_number = __builtin_bswap32(magic_number);

    uint32_t num_images_u32;
    if (gzread(file, &num_images_u32, sizeof(uint32_t)) != sizeof(uint32_t)) {
        fprintf(stderr, "Error: Failed to read number of images\n");
        gzclose(file);
        return NULL;
    }

    // Handle endianness
    num_images_u32 = __builtin_bswap32(num_images_u32);
    *num_images = (int)num_images_u32;

    uint32_t rows, cols;
    if (gzread(file, &rows, sizeof(uint32_t)) != sizeof(uint32_t) ||
        gzread(file, &cols, sizeof(uint32_t)) != sizeof(uint32_t)) {
        fprintf(stderr, "Error: Failed to read image dimensions\n");
        gzclose(file);
        return NULL;
    }

    // Handle endianness
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (magic_number != 0x00000803 || rows != 28 || cols != 28) {
        fprintf(stderr, "Error: Invalid MNIST image file format\n");
        gzclose(file);
        return NULL;
    }

    int** images = (int**)malloc(*num_images * sizeof(int*));
    if (images == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        gzclose(file);
        return NULL;
    }

    for (int i = 0; i < *num_images; i++) {
        images[i] = (int*)malloc(MNIST_IMAGE_SIZE * sizeof(int));
        if (images[i] == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            free_mnist_images_int(images, i);
            gzclose(file);
            return NULL;
        }
    }

    uint8_t* image_data = (uint8_t*)malloc(*num_images * MNIST_IMAGE_SIZE * sizeof(uint8_t));
    if (image_data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free_mnist_images_int(images, *num_images);
        gzclose(file);
        return NULL;
    }

    if (gzread(file, image_data, *num_images * MNIST_IMAGE_SIZE) != *num_images * MNIST_IMAGE_SIZE) {
        fprintf(stderr, "Error: Failed to read image data\n");
        free(image_data);
        free_mnist_images_int(images, *num_images);
        gzclose(file);
        return NULL;
    }

    // printf("First few bytes of image data: ");
    // for (int i = 0; i < 28*28; i++) {
    //     if (i % 28 == 0) {
    //         printf("\n");
    //     }
    //     printf("%02X ", image_data[i]);
    // }
    // printf("\n");

    gzclose(file);

    // Copy image data into 2D array in row-wise order
    int offset = 0;
    for (int i = 0; i < *num_images; i++) {
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            images[i][j] = image_data[offset++];
        }
    }

    free(image_data);
    return images;
}

uint8_t* load_mnist_labels(const char* file_path, int* num_labels) {
    gzFile file = gzopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_path);
        return NULL;
    }

    uint32_t magic_number;
    if (gzread(file, &magic_number, sizeof(uint32_t)) != sizeof(uint32_t)) {
        fprintf(stderr, "Error: Failed to read magic number\n");
        gzclose(file);
        return NULL;
    }

    // Handle endianness
    magic_number = __builtin_bswap32(magic_number);

    uint32_t num_labels_u32;
    if (gzread(file, &num_labels_u32, sizeof(uint32_t)) != sizeof(uint32_t)) {
        fprintf(stderr, "Error: Failed to read number of labels\n");
        gzclose(file);
        return NULL;
    }

    // Handle endianness
    num_labels_u32 = __builtin_bswap32(num_labels_u32);
    *num_labels = (int)num_labels_u32;

    if (magic_number != 0x00000801) {
        fprintf(stderr, "Error: Invalid MNIST label file format\n");
        gzclose(file);
        return NULL;
    }

    uint8_t* labels = (uint8_t*)malloc(*num_labels * sizeof(uint8_t));
    if (labels == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        gzclose(file);
        return NULL;
    }

    if (gzread(file, labels, *num_labels) != *num_labels) {
        fprintf(stderr, "Error: Failed to read label data\n");
        free(labels);
        gzclose(file);
        return NULL;
    }

    gzclose(file);
    return labels;
}

float** load_mnist_images_float(const char* file_path, int* num_images) {
    gzFile file = gzopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_path);
        return NULL;
    }

    uint32_t magic_number;
    if (gzread(file, &magic_number, sizeof(uint32_t)) != sizeof(uint32_t)) {
        fprintf(stderr, "Error: Failed to read magic number\n");
        gzclose(file);
        return NULL;
    }

    // Handle endianness
    magic_number = __builtin_bswap32(magic_number);

    uint32_t num_images_u32;
    if (gzread(file, &num_images_u32, sizeof(uint32_t)) != sizeof(uint32_t)) {
        fprintf(stderr, "Error: Failed to read number of images\n");
        gzclose(file);
        return NULL;
    }

    // Handle endianness
    num_images_u32 = __builtin_bswap32(num_images_u32);
    *num_images = (int)num_images_u32;

    uint32_t rows, cols;
    if (gzread(file, &rows, sizeof(uint32_t)) != sizeof(uint32_t) ||
        gzread(file, &cols, sizeof(uint32_t)) != sizeof(uint32_t)) {
        fprintf(stderr, "Error: Failed to read image dimensions\n");
        gzclose(file);
        return NULL;
    }

    // Handle endianness
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (magic_number != 0x00000803 || rows != 28 || cols != 28) {
        fprintf(stderr, "Error: Invalid MNIST image file format\n");
        gzclose(file);
        return NULL;
    }

    float** images = (float**)malloc(*num_images * sizeof(float*));
    if (images == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        gzclose(file);
        return NULL;
    }

    for (int i = 0; i < *num_images; i++) {
        images[i] = (float*)malloc(MNIST_IMAGE_SIZE * sizeof(float));
        if (images[i] == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            free_mnist_images(images, i, FLOAT32);
            gzclose(file);
            return NULL;
        }
    }

    uint8_t* image_data = (uint8_t*)malloc(*num_images * MNIST_IMAGE_SIZE * sizeof(uint8_t));
    if (image_data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free_mnist_images(images, *num_images, FLOAT32);
        gzclose(file);
        return NULL;
    }

    if (gzread(file, image_data, *num_images * MNIST_IMAGE_SIZE) != *num_images * MNIST_IMAGE_SIZE) {
        fprintf(stderr, "Error: Failed to read image data\n");
        free(image_data);
        free_mnist_images(images, *num_images, FLOAT32);
        gzclose(file);
        return NULL;
    }

    // printf("First few bytes of image data: ");
    // for (int i = 0; i < 28*28; i++) {
    //     if (i % 28 == 0) {
    //         printf("\n");
    //     }
    //     printf("%02X ", image_data[i]);
    // }

    gzclose(file);

    // Copy image data into 2D array and convert to float
    for (int i = 0; i < *num_images; i++) {
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            images[i][j] = (float)image_data[i * MNIST_IMAGE_SIZE + j] / 255.0f; // Normalize to [0, 1] range
        }
    }

    free(image_data);
    return images;
}

void free_mnist_images(void** images, int num_images, DataType data_type) {
    if (images == NULL) {
        return;
    }

    if (data_type == Int) {
        free_mnist_images_int((int**)images, num_images);
    } else {
        for (int i = 0; i < num_images; i++) {
            if (images[i] != NULL) {
                free(images[i]);
            }
        }
        free(images);
    }
}

void free_mnist_images_int(int** images, int num_images) {
    if (images == NULL) {
        return;
    }

    for (int i = 0; i < num_images; i++) {
        if (images[i] != NULL) {
            free(images[i]);
        }
    }

    free(images);
}

