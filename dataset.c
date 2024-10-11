//
//  dataset.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/16/24.
//

#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dirent.h>
#include <stdint.h>
#include <zlib.h>
#include <json-c/json.h>
#ifdef __unix__
    #include <libgen.h>  // Unix-based systems (Linux, macOS)
#elif defined(_WIN32)
    // Implementation of basename for Windows
    char *basename(char *path) {
        char *base = strrchr(path, '\\');  // Find the last backslash
        if (base) {
            return base + 1;  // Return the part after the last backslash
        }
        return path;  // If no backslash is found, the entire path is the basename
    }
#endif

static int load_images_from_json_array(Dataset* dataset, json_object* images_array, Dimensions input_dimension, DataType data_type) {

    int array_len = (int)json_object_array_length(images_array);

    // Check if dataset->images is NULL
    if (dataset->images == NULL) {
       // dataset->images is NULL, allocate memory
       dataset->images = (InputData**)malloc(array_len * sizeof(InputData*));
       if (dataset->images == NULL) {
           fprintf(stderr, "Error: Failed to allocate memory for images array\n");
           return -1;
       }

       // Initialize the images array with NULL pointers
       for (int i = 0; i < array_len; ++i) {
           dataset->images[i] = NULL;
       }
    }

    for (int index = 0; index < array_len; ++index) {
        json_object* image_obj = json_object_array_get_idx(images_array, index);
        if (image_obj != NULL) {
            json_object* file_path_obj = json_object_object_get(image_obj, "file_path");
            json_object* label_obj = json_object_object_get(image_obj, "label");
            if (file_path_obj != NULL && label_obj != NULL) {
                const char* file_path = json_object_get_string(file_path_obj);
                int label = json_object_get_int(label_obj);

                dataset->images[index] = load_input_data_from_image(file_path, &input_dimension, data_type);
                if (dataset->images[index] == NULL) {
                    fprintf(stderr, "Error: Failed to load image data from %s\n", file_path);
                    return -1;
                }
                dataset->labels[index] = label;
            }
        }
    }
    return 0;
}

Dataset* load_dataset_from_json(const char* file_path, Dimensions input_dimension, DataType data_type, int include_val_dataset) {
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
    if (json_buffer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    size_t bytes_read = fread(json_buffer, 1, file_size, file);
    if (bytes_read != file_size) {
        fprintf(stderr, "Error: Failed to read file content\n");
        free(json_buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    json_buffer[file_size] = '\0';

    // Parse JSON
    json_object* root = json_tokener_parse(json_buffer);
    free(json_buffer);
    json_buffer = NULL;
    if (root == NULL) {
        fprintf(stderr, "Error: Unable to parse JSON\n");
        return NULL;
    }

    // Calculate total steps for progress bar
    int total_steps = 5; // Initial steps
    json_object* num_images_obj = json_object_object_get(root, "num_images");
    int num_images = 0;
    if (num_images_obj != NULL) {
        num_images = json_object_get_int(num_images_obj);
        total_steps += num_images; // Each image is a step
    }
    if (include_val_dataset) {
        json_object* num_val_images_obj = json_object_object_get(root, "num_val_images");
        if (num_val_images_obj != NULL) {
            total_steps += json_object_get_int(num_val_images_obj); // Each validation image is a step
        }
    }

    int current_step = 0;

    // Allocate memory for Dataset struct
    Dataset* dataset = create_dataset(NULL, 0, input_dimension, data_type);
    if (dataset == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        json_object_put(root);
        return NULL;
    }

    // Read dataset information from JSON object
    json_object* dataset_name_obj = json_object_object_get(root, "dataset_name");
    if (dataset_name_obj != NULL) {
        const char* dataset_name = json_object_get_string(dataset_name_obj);
        dataset->name = strdup(dataset_name);
        if (dataset->name == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            free_dataset(dataset);
            json_object_put(root);
            return NULL;
        }
    }

    current_step++;
    print_progress_bar((float)current_step / total_steps, 50);

    if (num_images_obj != NULL) {
        dataset->num_images = json_object_get_int(num_images_obj);
        dataset->batch_size = dataset->num_images;

        // Allocate memory for images and labels
        dataset->images = (InputData**)malloc(dataset->num_images * sizeof(InputData*));
        dataset->labels = (int*)calloc(dataset->num_images, sizeof(int));
        if (dataset->images == NULL || dataset->labels == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            free_dataset(dataset);
            json_object_put(root);
            return NULL;
        }

        // Read images array
        json_object* images_array_obj = json_object_object_get(root, "images");
        if (images_array_obj != NULL) {
            if (load_images_from_json_array(dataset, images_array_obj, input_dimension, data_type) != 0) {
                free_dataset(dataset);
                json_object_put(root);
                return NULL;
            }
            current_step += dataset->num_images;
            print_progress_bar((float)current_step / total_steps, 50);
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
                dataset->val_dataset = create_dataset("val", num_val_images, input_dimension, data_type);
                if (dataset->val_dataset == NULL) {
                    fprintf(stderr, "Error: Memory allocation failed\n");
                    free_dataset(dataset);
                    json_object_put(root);
                    return NULL;
                }

                // Read validation images array
                json_object* val_images_array_obj = json_object_object_get(root, "val_images");
                if (val_images_array_obj != NULL) {
                    if (load_images_from_json_array(dataset->val_dataset, val_images_array_obj, input_dimension, data_type) != 0) {
                        free_dataset(dataset);
                        json_object_put(root);
                        return NULL;
                    }
                    current_step += dataset->val_dataset->num_images;
                    print_progress_bar((float)current_step / total_steps, 50);
                }
            }
        }
    } else {
        printf("\nSpliting the dataset...\n");
        total_steps = num_images;
        current_step = 0;
        json_object* validation_size_obj = json_object_object_get(root, "validation_size");
        if (validation_size_obj != NULL) {
            double validation_size = json_object_get_double(validation_size_obj);
            int num_val_images = (int)(dataset->num_images * validation_size);
            int num_train_images = dataset->num_images - num_val_images;

            // Split the dataset into training and validation sets
            Dataset* train_dataset = create_dataset(dataset->name, num_train_images, input_dimension, data_type);
            if (train_dataset == NULL) {
                fprintf(stderr, "Error: Memory allocation failed\n");
                free_dataset(dataset);
                json_object_put(root);
                return NULL;
            }

            dataset->val_dataset = create_dataset("val", num_val_images, input_dimension, data_type);
            if (dataset->val_dataset == NULL) {
                fprintf(stderr, "Error: Memory allocation failed\n");
                free(train_dataset->name);
                free(train_dataset);
                free_dataset(dataset);
                json_object_put(root);
                return NULL;
            }

            int train_index = 0;
            int val_index = 0;
            for (int i = 0; i < dataset->num_images; i++) {
                if (i < num_train_images) {
                    train_dataset->images[train_index] = dataset->images[i];
                    train_dataset->labels[train_index] = dataset->labels[i];
                    train_index++;
                    current_step++;
                    print_progress_bar((float)current_step / total_steps, 50);
                } else {
                    dataset->val_dataset->images[val_index] = dataset->images[i];
                    dataset->val_dataset->labels[val_index] = dataset->labels[i];
                    val_index++;
                    current_step++;
                    print_progress_bar((float)current_step / total_steps, 50);
                }
            }

            free(dataset->images);
            free(dataset->labels);
            dataset->images = train_dataset->images;
            dataset->labels = train_dataset->labels;
            dataset->batch_size = num_train_images;
            dataset->num_images = num_train_images;
            dataset->next_batch = NULL;
            free(train_dataset->name);
            train_dataset->name = NULL;
            train_dataset->images = NULL;
            train_dataset->labels = NULL;
            free(train_dataset);
            train_dataset = NULL;
        }
    }

    // Cleanup
    json_object_put(root);

    print_dataset_info(dataset);

    return dataset;

}

static int write_images_to_json(FILE* file, const char* folder_path, const char* label, bool* first_image) {
    DIR* dir = opendir(folder_path);
    if (dir == NULL) {
        fprintf(stderr, "Error: Unable to open folder '%s'\n", folder_path);
        return -1;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            fprintf(file, "%s    {\n", *first_image ? "" : ",");
            fprintf(file, "      \"file_path\": \"%s/%s\",\n", folder_path, entry->d_name);
            fprintf(file, "      \"label\": \"%s\"\n", label);
            fprintf(file, "    }\n");
            *first_image = false;
        }
    }
    closedir(dir);
    return 0;
}

void create_dataset_json_file(const char* folder_path, int include_val_dataset, float validation_size) {

    // Check if the folder path is valid
    if (folder_path == NULL || strlen(folder_path) == 0) {
        fprintf(stderr, "Error: Invalid folder path\n");
        return;
    }

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
            char subfolder_path[PATH_MAX];
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
                        char sub_subfolder_path[PATH_MAX];
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
                        } else {
                            fprintf(stderr, "Error: Unable to open sub-subfolder '%s'\n", sub_subfolder_path);
                        }
                    }
                }
                closedir(subfolder);
            } else {
                fprintf(stderr, "Error: Unable to open subfolder '%s'\n", subfolder_path);
            }
        }
    }

    // Construct the JSON file path
    char json_file_path[PATH_MAX];
    snprintf(json_file_path, sizeof(json_file_path), "%s/dataset.json", folder_path);

    // Open the JSON file for writing
    FILE* file = fopen(json_file_path, "w");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to create JSON file '%s'\n", json_file_path);
        closedir(dir);
        return;
    }

    // Write dataset information to JSON file
    char* path = malloc((2 + strlen(folder_path)) * sizeof(char));
    if (!path) {
        fprintf(stderr, "Copy dataset_name (folder path) failed!");
    }
    strcpy(path, folder_path);
    fprintf(file, "{\n");
    fprintf(file, "  \"dataset_name\": \"%s\",\n", basename(path));
    fprintf(file, "  \"num_images\": %d,\n", num_images);
    fprintf(file, "  \"images\": [\n");

    // Reset the directory stream pointer to the beginning
    rewinddir(dir);

    // Write image information to JSON file
    bool first_image = true;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 && strcmp(entry->d_name, "val") != 0) {
            char subfolder_path[PATH_MAX];
            snprintf(subfolder_path, sizeof(subfolder_path), "%s/%s", folder_path, entry->d_name);
            if (write_images_to_json(file, subfolder_path, entry->d_name, &first_image) != 0) {
                free(path);
                fclose(file);
                closedir(dir);
                return;
            }
        }
    }

    // Close the JSON array
    fprintf(file, "  ]\n");

    // Write validation dataset information if include_val_dataset is true
    if (include_val_dataset && num_val_images > 0) {
        fprintf(file, ",\n  \"val_dataset\": \"%s\",\n", "val");
        fprintf(file, "  \"num_val_images\": %d,\n", num_val_images - 2);
        fprintf(file, "  \"val_images\": [\n");

        // Reset the directory stream pointer to the beginning
        rewinddir(dir);

        bool first_val_image = true;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 && strcmp(entry->d_name, "val") == 0) {
                char subfolder_path[PATH_MAX];
                snprintf(subfolder_path, sizeof(subfolder_path), "%s/%s", folder_path, entry->d_name);
                DIR* subfolder = opendir(subfolder_path);
                if (subfolder != NULL) {
                    struct dirent* subentry;
                    while ((subentry = readdir(subfolder)) != NULL) {
                        if (subentry->d_type == DT_DIR && strcmp(subentry->d_name, "..") != 0) {
                            char sub_subfolder_path[PATH_MAX];
                            snprintf(sub_subfolder_path, sizeof(sub_subfolder_path), "%s/%s", subfolder_path, subentry->d_name);
                            if (write_images_to_json(file, sub_subfolder_path, subentry->d_name, &first_val_image) != 0) {
                                free(path);
                                fclose(file);
                                closedir(dir);
                                return;
                            }
                        }
                    }
                    closedir(subfolder);
                } else {
                    fprintf(stderr, "Error: Unable to open subfolder '%s'\n", subfolder_path);
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
    free(path);
    fclose(file);
    closedir(dir);

    printf("Create JSON file of '%s' successfully!\n", json_file_path);
}

Dataset* create_dataset(const char* name, int num_images, Dimensions input_dimensions, DataType data_type) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (dataset == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    if (name != NULL) {
        dataset->name = strdup(name);
        if (dataset->name == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            free(dataset);
            dataset = NULL;
            return NULL;
        }
    }

    dataset->batch_size = num_images;
    dataset->num_images = num_images;
    dataset->data_dimensions = input_dimensions;
    dataset->data_type = data_type;

    if (num_images != 0) {
        dataset->images = (InputData**)malloc(num_images * sizeof(InputData*));
        dataset->labels = (int*)calloc(num_images, sizeof(int));
        if (dataset->images == NULL || dataset->labels == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            free(dataset->name);
            dataset->name = NULL;
            free(dataset->images);
            dataset->images = NULL;
            free(dataset->labels);
            dataset->labels = NULL;
            free(dataset);
            dataset = NULL;
            return NULL;
        }
    }

    dataset->next_batch = NULL;
    dataset->val_dataset = NULL;

    return dataset;
}

Dataset* split_dataset_into_batches(Dataset* original_dataset, int num_batches) {
    // Calculate the number of images per batch
    int images_per_batch = original_dataset->num_images / num_batches;
    int remaining_images = original_dataset->num_images % num_batches;
    if (images_per_batch == 0) {
        fprintf(stderr, "Error: Insufficient number of images for the specified number of batches\n");
        return NULL;
    }

    // Create the head of the linked list
    Dataset* head = NULL;
    Dataset* tail = NULL;

    int start_index = 0;
    for (int i = 0; i < num_batches; i++) {
        // Calculate the number of images in this batch
        int batch_size = images_per_batch + (i < remaining_images ? 1 : 0);

        // Allocate memory for the batch dataset
        Dataset* batch = create_dataset(original_dataset->name, batch_size, original_dataset->data_dimensions, original_dataset->data_type);
        if (batch == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            // Free previously allocated batches
            free_dataset(head);
            return NULL;
        }

        // Copy dataset information to the batch dataset
        if (i == 0) {
            batch->val_dataset = copy_dataset(original_dataset->val_dataset);
            if (batch->val_dataset == NULL) {
                fprintf(stderr, "Copy val_dataset for batches failed\n");
                free_dataset(head);
                free_dataset(batch);
                return NULL;
            }
        } else {
            batch->val_dataset = head->val_dataset;
        }

        // Copy images and labels to the batch dataset
        for (int j = 0; j < batch_size; j++) {
            batch->images[j] = copy_image_data(original_dataset->images[start_index + j], original_dataset->data_dimensions, original_dataset->data_type);
            if (batch->images[j] == NULL) {
                fprintf(stderr, "Error: Failed to copy image data\n");
                // Free previously allocated batches (including the current one)
                free_dataset(head);
                free_dataset(batch);
                return NULL;
            }
        }
        if (original_dataset->labels != NULL) {
            memcpy(batch->labels, &original_dataset->labels[start_index], batch_size * sizeof(int));
        }


        // Update start index for the next batch
        start_index += batch_size;

        // Add the batch to the linked list
        if (head == NULL) {
            head = batch;
            tail = batch;
        } else {
            tail->next_batch = batch;
            tail = batch;
        }
    }

    return head;
}

void print_dataset_info(Dataset* dataset) {
    if (dataset == NULL) {
        printf("The dataset is NULL\n");
        return;
    }

    printf("\nLoaded dataset:\n");
    printf("  Name:                    %s\n", dataset->name);
    printf("  Dimensions:              %d x %d x %d\n", dataset->data_dimensions.width, dataset->data_dimensions.height, dataset->data_dimensions.channels);
    printf("  Data Type:               %s\n", dataset->data_type == SINT32? "UINT8" : "FLOAT32");

    if (dataset->val_dataset != NULL) {
        printf("  Training Set Size:       %d\n", dataset->num_images);
        printf("  Validation Dataset Name: %s\n", dataset->val_dataset->name);
        printf("  Number of Images:        %d\n", dataset->val_dataset->num_images);
        printf("  Input Dimensions:        %dx%d, Channels: %d\n", dataset->val_dataset->data_dimensions.width, dataset->val_dataset->data_dimensions.height, dataset->val_dataset->data_dimensions.channels);
        printf("  Data Type:               %s\n", dataset->val_dataset->data_type == SINT32 ? "SINT32" : "FLOAT32");
        printf("  Test Set Size:           %d\n", dataset->val_dataset->num_images);
    } else {
        printf("  Training Set Size:       %d\n", dataset->num_images);
    }
}

Dataset* copy_dataset(Dataset* original_dataset) {
    if (original_dataset == NULL) {
        return NULL;
    }

    Dataset* dataset = create_dataset(original_dataset->name, original_dataset->num_images, original_dataset->data_dimensions, original_dataset->data_type);
    if (dataset == NULL) {
        fprintf(stderr, "Allocating new dataset failed\n");
        return NULL;
    }

    Dataset* temp = dataset;
    Dataset* current = original_dataset;
    while (current != NULL) {
        Dataset* next = current->next_batch;

        if (current->images != NULL) {
            for (int i = 0; i < current->num_images; ++i) {
                temp->images[i] = copy_image_data(current->images[i], current->data_dimensions, current->data_type);
                if (temp->images[i] == NULL) {
                    fprintf(stderr, "Copy new images failed\n");
                    free_dataset(dataset);
                    return NULL;
                }
            }
        }

        if (current->labels != NULL) {
            memcpy(temp->labels, current->labels, current->num_images * sizeof(int));
        }

        if (current->val_dataset != NULL && current->val_dataset != current) {
            temp->val_dataset = copy_dataset(current->val_dataset);
            if (temp->val_dataset == NULL) {
                fprintf(stderr, "Copy val_dataset failed\n");
                free_dataset(dataset);
                return NULL;
            }
        }

        if (current->next_batch != NULL) {
            temp->next_batch = create_dataset(next->name, next->num_images, next->data_dimensions, next->data_type);
            if (temp->next_batch == NULL) {
                fprintf(stderr, "Copy next batch failed\n");
                free_dataset(dataset);
                return NULL;
            }
        }

        temp = temp->next_batch;
        current = next;
    }

    return dataset;
}

void free_dataset(Dataset* dataset) {
    if (dataset == NULL) {
        printf("Dataset is NULL\n");
        return;
    }

    Dataset* current = dataset;
    while (current != NULL) {
        Dataset* next = current->next_batch;

        if (current->images != NULL) {
            for (int i = 0; i < current->num_images; ++i) {
                free_image_data(current->images[i], current->data_dimensions, current->data_type);
                current->images[i] = NULL;
            }
            free(current->images);
            current->images = NULL;
        }

        if (current->labels != NULL) {
            free(current->labels);
            current->labels = NULL;
        }

        if (current->name != NULL) {
            free(current->name);
            current->name = NULL;
        }

        current->num_images = 0;
        current->batch_size = 0;

        if (current->val_dataset != NULL && current->val_dataset != current) {
            free_dataset(current->val_dataset);
            Dataset* temp = next;
            while (temp != NULL && temp->val_dataset != NULL) {
                temp->val_dataset = NULL;
                temp = temp->next_batch;
            }
        }
        current->val_dataset = NULL;

        Dataset* temp = current;
        current = next;
        free(temp);
        temp = NULL;
    }
}

static void free_mnist_images(void** images, DataType type) {
    switch (type) {
        case SINT32:
            free_2d_int_array((int**)images);
            break;
        case FLOAT32:
            free_2d_float_array((float**)images);
            break;;
        default:
            fprintf(stderr, "Unvalid type\n");
            break;
    }
}

static int** load_mnist_images_int(const char* file_path, int* num_images) {
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

    if (magic_number != 0x00000803 || rows != MNIST_IMAGE_WIDTH || cols != MNIST_IMAGE_HEIGHT) {
        fprintf(stderr, "Error: Invalid MNIST image file format\n");
        gzclose(file);
        return NULL;
    }

    int** images = calloc_2d_int_array(*num_images, MNIST_IMAGE_SIZE);
    if (images == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        gzclose(file);
        return NULL;
    }

    uint8_t* image_data = (uint8_t*)malloc(*num_images * MNIST_IMAGE_SIZE * sizeof(uint8_t));
    if (image_data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free_2d_int_array(images);
        gzclose(file);
        return NULL;
    }

    if (gzread(file, image_data, *num_images * MNIST_IMAGE_SIZE) != *num_images * MNIST_IMAGE_SIZE) {
        fprintf(stderr, "Error: Failed to read image data\n");
        free(image_data);
        image_data = NULL;
        free_2d_int_array(images);
        gzclose(file);
        return NULL;
    }

    gzclose(file);

    // Copy image data into 2D array in row-wise order
    int offset = 0;
    for (int i = 0; i < *num_images; i++) {
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            images[i][j] = (int)image_data[offset++];
        }
    }

    free(image_data);
    image_data = NULL;
    return images;
}

static uint8_t* load_mnist_labels(const char* file_path, int* num_labels) {
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
        labels = NULL;
        gzclose(file);
        return NULL;
    }

    gzclose(file);
    return labels;
}

static float** load_mnist_images_float(const char* file_path, int* num_images) {
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

    if (magic_number != 0x00000803 || rows != MNIST_IMAGE_WIDTH || cols != MNIST_IMAGE_HEIGHT) {
        fprintf(stderr, "Error: Invalid MNIST image file format\n");
        gzclose(file);
        return NULL;
    }

    float** images = calloc_2d_float_array(*num_images, MNIST_IMAGE_SIZE);
    if (images == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        gzclose(file);
        return NULL;
    }

    uint8_t* image_data = (uint8_t*)malloc(*num_images * MNIST_IMAGE_SIZE * sizeof(uint8_t));
    if (image_data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free_2d_float_array(images);
        gzclose(file);
        return NULL;
    }

    if (gzread(file, image_data, *num_images * MNIST_IMAGE_SIZE) != *num_images * MNIST_IMAGE_SIZE) {
        fprintf(stderr, "Error: Failed to read image data\n");
        free(image_data);
        image_data = NULL;
        free_2d_float_array(images);
        gzclose(file);
        return NULL;
    }

    gzclose(file);

    // Copy image data into 2D array and convert to float
    for (int i = 0; i < *num_images; i++) {
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            images[i][j] = (float)image_data[i * MNIST_IMAGE_SIZE + j] / 255.0f; // Normalize to [0, 1] range
        }
    }

    free(image_data);
    image_data = NULL;
    return images;
}

Dataset* load_mnist_dataset(const char* train_images_path, const char* train_labels_path,
                             const char* test_images_path, const char* test_labels_path,
                             DataType data_type) {
    // Load training data
    int num_train_images, num_train_labels;
    void** train_images = NULL;
    uint8_t* train_labels = NULL;

    if (data_type == SINT32) {
        train_images = (void**)load_mnist_images_int(train_images_path, &num_train_images);
        train_labels = load_mnist_labels(train_labels_path, &num_train_labels);
    } else if (data_type == FLOAT32) {
        train_images = (void**)load_mnist_images_float(train_images_path, &num_train_images);
        train_labels = load_mnist_labels(train_labels_path, &num_train_labels);
    } else {
        fprintf(stderr, "Error: Unsupported data type\n");
        return NULL;
    }

    if (train_images == NULL || train_labels == NULL || num_train_images != num_train_labels) {
        fprintf(stderr, "Error: Failed to load training data\n");
        free_mnist_images(train_images, data_type);
        free(train_labels);
        train_labels = NULL;
        return NULL;
    }

    Dimensions input_dimension = {MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_CHANNEL};

    Dataset* dataset = create_dataset("MNIST", num_train_images, input_dimension, data_type);
    if (dataset == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free_mnist_images(train_images, data_type);
        free(train_labels);
        train_labels = NULL;
        free_dataset(dataset);
        return NULL;
    }

    printf("\nLoading training data: ");
    for (int i = 0; i < num_train_images; ++i) {
        dataset->images[i] = (InputData*)malloc(sizeof(InputData));
        if (data_type == SINT32) {
            dataset->images[i]->int_data = calloc_3d_int_array(MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_CHANNEL);
            int* int_data_p = &dataset->images[i]->int_data[0][0][0];
            memcpy(int_data_p, (int*)train_images[i], MNIST_IMAGE_SIZE * sizeof(int));
        } else if (data_type == FLOAT32) {
            dataset->images[i]->float32_data = calloc_3d_float_array(MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_CHANNEL);
            float* float_data_p = &dataset->images[i]->float32_data[0][0][0];
            memcpy(float_data_p, (float*)train_images[i], MNIST_IMAGE_SIZE * sizeof(float));
        }
        dataset->labels[i] = (int)train_labels[i]; // Cast to int
        print_progress_bar((float)(i + 1) / num_train_images, 50);
    }
    printf("\n");

    free(train_labels);
    train_labels = NULL;
    free_mnist_images(train_images, data_type);
    train_images = NULL;

    // Load test data
    int num_test_images, num_test_labels;
    void** test_images = NULL;
    uint8_t* test_labels = NULL;

    if (data_type == SINT32) {
        test_images = (void**)load_mnist_images_int(test_images_path, &num_test_images);
        test_labels = load_mnist_labels(test_labels_path, &num_test_labels);
    } else {
        test_images = (void**)load_mnist_images_float(test_images_path, &num_test_images);
        test_labels = load_mnist_labels(test_labels_path, &num_test_labels);
    }

    if (test_images == NULL || test_labels == NULL || num_test_images != num_test_labels) {
        fprintf(stderr, "Error: Failed to load test data\n");
        free_mnist_images(test_images, num_test_images);
        free(test_labels);
        test_labels = NULL;
        free_dataset(dataset);
        return NULL;
    }

    dataset->val_dataset = create_dataset("MNIST Test", num_test_images, input_dimension, data_type);

    if (dataset->val_dataset == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free_mnist_images(test_images, num_test_images);
        free(test_labels);
        test_labels = NULL;
        free_dataset(dataset);
        return NULL;
    }

    printf("Loading test data: ");
    for (int i = 0; i < num_test_images; i++) {
        dataset->val_dataset->images[i] = (InputData*)malloc(sizeof(InputData));
        if (data_type == SINT32) {
            dataset->val_dataset->images[i]->int_data = calloc_3d_int_array(MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_CHANNEL);
            int* int_val_data_p = &dataset->val_dataset->images[i]->int_data[0][0][0];
            memcpy(int_val_data_p, (int*)test_images[i], MNIST_IMAGE_SIZE * sizeof(int));
        } else if (data_type == FLOAT32) {
            dataset->val_dataset->images[i]->float32_data = calloc_3d_float_array(MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_CHANNEL);
            float* float_val_data_p = &dataset->val_dataset->images[i]->float32_data[0][0][0];
            memcpy(float_val_data_p, (float*)test_images[i], MNIST_IMAGE_SIZE * sizeof(float));
        }
        dataset->val_dataset->labels[i] = (int)test_labels[i]; // Cast to int
        print_progress_bar((float)(i + 1) / num_test_images, 50);
    }

    printf("\nFinished loading MNIST Dataset!\n");

    free(test_labels);
    test_labels = NULL;
    free_mnist_images(test_images, data_type);
    test_images = NULL;

    print_dataset_info(dataset);

    return dataset;
}
