//
//  main.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/14/24.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <unistd.h>

#include "input/data.h"
#include "dataset.h"
#include "utils/utils.h"
#include "model/layer/layer.h"
#include "model/model.h"
#include "cdnn.h"

// #include "input/data.c"
// #include "model/layer/layer.c"
// #include "model/loss/losses.c"
// #include "model/metric/metric.c"
// #include "model/optimizer/optimizer.c"
// #include "model/model.c"
// #include "utils/compute/activation.c"
// #include "utils/compute/convolution.c"
// #include "utils/compute/dropout.c"
// #include "utils/compute/flatten.c"
// #include "utils/compute/fully_connected.c"
// #include "utils/compute/loss_function.c"
// #include "utils/compute/metric_algorithm.c"
// #include "utils/compute/optim_algorithm.c"
// #include "utils/compute/pooling.c"
// #include "utils/memory.c"
// #include "utils/tools.c"
// #include "dataset.c"
// #include "cdnn.c"

#define CURRENT_FILE_PATH __FILE__

// TODO: Define input dimensions
#define INPUT_WIDTH    224
#define INPUT_HEIGHT   224
#define INPUT_CHANNELS 3

int main(int argc, const char * argv[]) {
    // insert code here...
    printf("Hello, World!\n");

    printf("Current Position: %s\n", PROJECT_PATH);
    char fullpath[2048];

    // TODO: - Test `data.h`
    char rel_path[1024] = "dataset example/test_data_and_val/0/test.jpeg";
    snprintf(fullpath, sizeof(fullpath), "%s/%s", PROJECT_PATH, rel_path);
    Dimensions new_dimensions;
    new_dimensions.width    = 100;
    new_dimensions.height   = 100;
    new_dimensions.channels = 3;
    InputData *image_data = load_input_data_from_image(fullpath, &new_dimensions, FLOAT32);
    if (image_data == NULL) {
        fprintf(stderr, "Error: Failed to load the image.\n");
        return 1;
    }

    printf("Loading Successful!\n");

    // print the first two line
    for (int i = 66; i < 70; i++) {
        for (int j = 88; j < 91; j++) {
            printf("%f ", image_data->float32_data[i][j][0]);
        }
        printf("\n");
    }

    // Free the memory allocated for image data
    free_image_data(image_data, new_dimensions, INT);

    printf("Image processing completed successfully.\n");

    // TODO: - Test `dataset.h`
    // Define input dimensions
    Dimensions input_dimensions = {INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS}; // Example dimensions

    strcpy(rel_path, "dataset example/test_data_without_val");
    snprintf(fullpath, sizeof(fullpath), "%s/%s", PROJECT_PATH, rel_path);
    create_dataset_json_file(fullpath, 0, 0.5);

    // Load dataset from JSON file
    printf("\nLoading dataset from JSON file...\n");
    strcpy(rel_path, "dataset example/test_data_without_val/dataset.json");
    snprintf(fullpath, sizeof(fullpath), "%s/%s", PROJECT_PATH, rel_path);
    Dataset* dataset = load_dataset_from_json(fullpath, input_dimensions, INT, 0);
    if (dataset == NULL) {
        fprintf(stderr, "Error: Failed to load dataset from JSON file\n");
        return 1;
    }
    printf("Dataset loaded successfully\n");

    // Create a new JSON file from dataset
    printf("Creating JSON file from dataset...\n");
    strcpy(rel_path, "dataset example/test_data_and_val");
    snprintf(fullpath, sizeof(fullpath), "%s/%s", PROJECT_PATH, rel_path);
    create_dataset_json_file(fullpath, 1, 0);
    printf("JSON file created successfully!\n");

    // Split dataset into batches
    int num_batches = 2; // Example number of batches
    Dataset* batched_dataset = split_dataset_into_batches(dataset, num_batches);
    free_dataset(dataset);
    dataset = batched_dataset;
    if (dataset == NULL) {
        fprintf(stderr, "Error: Failed to split dataset into batches\n");
        return 1;
    }
    printf("Dataset split into %d batches successfully\n", num_batches);

    // Free dataset memory
    free_dataset(dataset);

    // TODO: - Test MNIST Loading
//    const char* train_images_path = "dataset example/mnist/train-images-idx3-ubyte.gz";
//    const char* train_labels_path = "dataset example/mnist/train-labels-idx1-ubyte.gz";
//    const char* test_images_path  = "dataset example/mnist/t10k-images-idx3-ubyte.gz";
//    const char* test_labels_path  = "dataset example/mnist/t10k-labels-idx1-ubyte.gz";
//
//    Dataset* mnist_dataset = load_mnist_dataset(train_images_path, train_labels_path,
//                                                 test_images_path, test_labels_path, FLOAT32);
//    if (mnist_dataset == NULL) {
//        fprintf(stderr, "Error: Failed to load MNIST dataset\n");
//        return 1;
//    }
//
//    // Free the dataset
//    free_dataset(mnist_dataset);

    // TODO: Test the `layer.h`

//    // Create a convolutional layer
//    LayerParams conv_params;
//    conv_params.conv_params.num_filters = 256;
//    conv_params.conv_params.filter_size = 3;
//    conv_params.conv_params.stride      = 1;
//    conv_params.conv_params.padding     = 1;
//    conv_params.conv_params.activation  = RELU;
//
//    Dimensions input_shape = {224, 224, 1};
//    Layer* conv_layer = create_layer(CONVOLUTIONAL, conv_params);
//    conv_layer->input_shape = input_shape;
//    initialize_layer(conv_layer);
//    compute_output_shape(conv_layer);
//
//    printf("Convolutional Layer:\n");
//    printf("Input  Shape: (%d, %d, %d)\n", conv_layer->input_shape.width,  conv_layer->input_shape.height,  conv_layer->input_shape.channels);
//    printf("Output Shape: (%d, %d, %d)\n", conv_layer->output_shape.width, conv_layer->output_shape.height, conv_layer->output_shape.channels);
//
//    // Create an activation layer after the convolutional layer
//    LayerParams activation_params;
//    activation_params.activation_params.activation = RELU;
//    Layer* activation_layer = create_layer(ACTIVATION, activation_params);
//    activation_layer->input_shape = conv_layer->output_shape;
//    compute_output_shape(activation_layer);
//
//    // Link the layers
//    conv_layer->next_layer = activation_layer;
//    activation_layer->prev_layer = conv_layer;
//
//    // Create a pooling layer
//    LayerParams pool_params;
//    pool_params.pooling_params.pool_size = 2;
//    pool_params.pooling_params.stride    = 2;
//    pool_params.pooling_params.pool_type = MAX;
//
//    Layer* pool_layer = create_layer(POOLING, pool_params);
//    pool_layer->input_shape = activation_layer->output_shape;
//    compute_output_shape(pool_layer);
//
//    printf("\nPooling Layer:\n");
//    printf("Input  Shape: (%d, %d, %d)\n", pool_layer->input_shape.height, pool_layer->input_shape.width, pool_layer->input_shape.channels);
//    printf("Output Shape: (%d, %d, %d)\n", pool_layer->output_shape.height, pool_layer->output_shape.width, pool_layer->output_shape.channels);
//
//    // Link the layers
//    activation_layer->next_layer = pool_layer;
//    pool_layer->prev_layer = activation_layer;
//
//    // Create a fully connected layer
//    LayerParams fc_params;
//    fc_params.fc_params.num_neurons = 10;
//    fc_params.fc_params.activation  = SOFTMAX;
//
//    Layer* fc_layer = create_layer(FULLY_CONNECTED, fc_params);
//    fc_layer->input_shape = pool_layer->output_shape;
//    initialize_layer(fc_layer);
//    compute_output_shape(fc_layer);
//
//    printf("\nFully Connected Layer:\n");
//    printf("Input  Shape: (%d, %d, %d)\n", fc_layer->input_shape.height, fc_layer->input_shape.width, fc_layer->input_shape.channels);
//    printf("Output Shape: (%d, %d, %d)\n", fc_layer->output_shape.height, fc_layer->output_shape.width, fc_layer->output_shape.channels);
//
//    // Link the layers
//    pool_layer->next_layer = fc_layer;
//    fc_layer->prev_layer = pool_layer;
//
//    // Allocate memory for input and output tensors
//    float*** input_tensor = calloc_3d_float_array(input_shape.height, input_shape.width, input_shape.channels);
//    float*** conv_output;
//    float*** activation_output;
//    float*** pool_output;
//    float*** fc_output;
//
//    // Forward pass
//    conv_output       = layer_forward_pass(conv_layer, input_tensor,1);
//    activation_output = layer_forward_pass(activation_layer, conv_output,1);
//    pool_output       = layer_forward_pass(pool_layer, activation_output,1);
//    fc_output         = layer_forward_pass(fc_layer, pool_output,1);
//
//    // Allocate memory for gradients
//    float*** conv_input_grad       = calloc_3d_float_array(input_shape.height, input_shape.width, input_shape.channels);
//    float*** activation_input_grad = calloc_3d_float_array(activation_layer->input_shape.height, activation_layer->input_shape.width, activation_layer->input_shape.channels);
//    float*** pool_input_grad       = calloc_3d_float_array(pool_layer->input_shape.height, pool_layer->input_shape.width, pool_layer->input_shape.channels);
//    float*** fc_input_grad         = calloc_3d_float_array(fc_layer->input_shape.height, fc_layer->input_shape.width, fc_layer->input_shape.channels);
//
//    float*** conv_output_grad       = calloc_3d_float_array(conv_layer->output_shape.height, conv_layer->output_shape.width, conv_layer->output_shape.channels);
//    float*** activation_output_grad = calloc_3d_float_array(activation_layer->output_shape.height, activation_layer->output_shape.width, activation_layer->output_shape.channels);
//    float*** pool_output_grad       = calloc_3d_float_array(pool_layer->output_shape.height, pool_layer->output_shape.width, pool_layer->output_shape.channels);
//    float*** fc_output_grad         = calloc_3d_float_array(1, 1, fc_layer->output_shape.channels);
//
//    // Backward pass
//    layer_backward_pass(fc_layer,         fc_output_grad,         fc_input_grad);
//    layer_backward_pass(pool_layer,       pool_output_grad,       pool_input_grad);
//    layer_backward_pass(activation_layer, activation_output_grad, activation_input_grad);
//    layer_backward_pass(conv_layer,       conv_output_grad,       conv_input_grad);
//
//    // Clean up
//    free_3d_float_array(input_tensor);
//    free_3d_float_array(conv_output);
//    free_3d_float_array(activation_output);
//    free_3d_float_array(pool_output);
//    free_3d_float_array(fc_output);
//
//    free_3d_float_array(conv_input_grad);
//    free_3d_float_array(activation_input_grad);
//    free_3d_float_array(pool_input_grad);
//    free_3d_float_array(fc_input_grad);
//
//    free_3d_float_array(conv_output_grad);
//    free_3d_float_array(activation_output_grad);
//    free_3d_float_array(pool_output_grad);
//    free_3d_float_array(fc_output_grad);
//
//    delete_layer(conv_layer);
//    delete_layer(activation_layer);
//    delete_layer(pool_layer);
//    delete_layer(fc_layer);

    // TODO: - Test `model.h` file

    // Define input and output dimensions
    Dimensions input_dim = {28, 28, 3};
    Dimensions output_dim = {1, 1, 10};

    // Create a new model
    Model* model = create_model(input_dim, output_dim);

    add_layer(model, CONVOLUTIONAL,   32, 3, 1, 1, RELU,    0,   0.0f);
    add_layer(model, POOLING,         0,  2, 2, 0, 0,       MAX, 0.0f);
    add_layer(model, FULLY_CONNECTED, 10, 0, 0, 0, SOFTMAX, 0,   0.0f);

    // Set optimizer, loss function, and metric
    compile_model(model, SGD, 0.01f, MSE, ACCURACY);

    // Load dataset
    strcpy(rel_path, "dataset example/test_data_and_val/dataset.json");
    snprintf(fullpath, sizeof(fullpath), "%s/%s", PROJECT_PATH, rel_path);
    dataset = load_dataset_from_json(fullpath, input_dim, FLOAT32, 1);
    if (dataset == NULL) {
        fprintf(stderr, "Error: Failed to load dataset\n");
        free_model(model);
        return 1;
    }

    // Split dataset into batches
    Dataset* splited_dataset = split_dataset_into_batches(dataset, 2);
    if (splited_dataset == NULL) {
        fprintf(stderr, "Error: Failed to split dataset into batches\n");
        delete_model(model);
        free_dataset(dataset);
        return 1;
    }

    // Train the model
    int num_epochs = 10;
    // // print some data to the console
    // printf("input data: %d %d %d\n", dataset->images[0]->data.float32[0][0][0], dataset->images[3][0][1], dataset->images[27][0][2]);
    train_model(model, splited_dataset, num_epochs);

    // Evaluate the model
    float accuracy = evaluate_model(model, splited_dataset);
    printf("Model accuracy: %.2f%%\n", accuracy * 100);

    // Clean up
    free_dataset(dataset);
    free_dataset(splited_dataset);
    delete_model(model);

    // TODO: - Test the APIs

    // Load pre-trained model
    ModelConfig vgg16_config = {"Adam", 0.0003f, "categorical_crossentropy", "accuracy"};
    strcpy(rel_path, "VGG16 weights.h5");
    snprintf(fullpath, sizeof(fullpath), "%s/%s", PROJECT_PATH, rel_path);
    Model *vgg16 = load_vgg16(fullpath, 1, 1000, vgg16_config);
    if (vgg16 == NULL) {
        printf("Error loading VGG16 model\n");
        return 1;
    }
    strcpy(rel_path, "vgg16_weights.h5");
    snprintf(fullpath, sizeof(fullpath), "%s/%s", PROJECT_PATH, rel_path);
    save_weights(vgg16, fullpath);
    free_model(vgg16);

    // Or Load MNIST dataset directly
    const char* train_images_rel = "/dataset example/mnist/train-images-idx3-ubyte.gz";
    const char* train_labels_rel = "/dataset example/mnist/train-labels-idx1-ubyte.gz";
    const char* test_images_rel  = "/dataset example/mnist/t10k-images-idx3-ubyte.gz";
    const char* test_labels_rel  = "/dataset example/mnist/t10k-labels-idx1-ubyte.gz";

    char train_images_path[2048];
    char train_labels_path[2048];
    char test_images_path[2048];
    char test_labels_path[2048];

    snprintf(train_images_path, sizeof(train_images_path), "%s%s", PROJECT_PATH, train_images_rel);
    snprintf(train_labels_path, sizeof(train_labels_path), "%s%s", PROJECT_PATH, train_labels_rel);
    snprintf(test_images_path, sizeof(test_images_path), "%s%s", PROJECT_PATH, test_images_rel);
    snprintf(test_labels_path, sizeof(test_labels_path), "%s%s", PROJECT_PATH, test_labels_rel);

    dataset = load_mnist_dataset(train_images_path, train_labels_path,
                                 test_images_path, test_labels_path, FLOAT32);
    if (dataset == NULL) {
        printf("Error loading dataset\n");
        return 1;
    }

    // Split dataset into batches
    splited_dataset = split_dataset_into_batches(dataset, 1875);
    if (dataset == NULL) {
        printf("Error splitting dataset into batches\n");
        free_dataset(dataset);
        return 1;
    }

    // Create a new model
    model = create(28, 28, 1, 1, 1, 10);
    if (model == NULL) {
        printf("Error creating model\n");
        free_dataset(dataset);
        free_dataset(splited_dataset);
        return 1;
    }

    // Add layers to the model
    add_convolutional_layer(model, 3, 3, 1, 1, "relu");
    add_pooling_layer(model, 2, 2, "max");
    add_fully_connected_layer(model, 128, "relu");
    add_dropout_layer(model, 0.5f);
    add_fully_connected_layer(model, 10, "softmax");

    // Compile the model
    ModelConfig config = {"sgd", 0.001f, "categorical_crossentropy", "accuracy"};
    compile(model, config);

    // Train the model
    train(model, splited_dataset, 3);

    // Evaluate the model
    accuracy = evaluate(model, dataset->val_dataset);
    printf("FInal Validation Accuracy: %.f%%\n", accuracy * 100.0f);

    // Save the model
    strcpy(rel_path, "test_model_config.json");
    snprintf(fullpath, sizeof(fullpath), "%s/%s", PROJECT_PATH, rel_path);
    int result = save_model_to_json(model, fullpath);
    if (result != 0) {
        printf("Error saving model\n");
    }

    strcpy(rel_path, "model_config.json");
    snprintf(fullpath, sizeof(fullpath), "%s/%s", PROJECT_PATH, rel_path);
    Model* model2 = create_model_from_json(fullpath);

    // Free memory
    free_model(model);
    free_model(model2);
    free_dataset(dataset);
    free_dataset(splited_dataset);

    return 0;
}

