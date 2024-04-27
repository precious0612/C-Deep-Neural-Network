/* dataset.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "dataset.h"

#include <json-c/json.h>

// Define the maximum number of images per batch
#define MAX_IMAGES_PER_BATCH 100

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

Dataset* split_dataset_into_batches(Dataset* dataset, int num_batches) {
    Dataset** batches = create_batches(dataset, num_batches);
    if (batches == NULL) {
        return NULL;
    }

    batches[0]->val_dataset = dataset->val_dataset;
    
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

    dataset = batches[0];
    for (int i = 1; i < num_batches; i++) {
        dataset->next_batch = batches[i];
    }

    free(batches);

    return dataset;
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
