/* CNN.h (high-level) */

#ifndef CNN_H
#define CNN_H

#include "model/model.h"  // Include model.h to access Model structure

// Define structure for holding model configuration
typedef struct {
    // int input_shape[3];       // Shape of the input data (width, height, channels)
    // int output_shape[3];      // Shape of the output data (width, height, channels)
    char optimizer[20];       // Name of the optimizer (e.g., "SGD", "Adam")
    float learning_rate;      // Learning rate for optimization
    char loss_function[30];   // Name of the loss function (e.g., "categorical_crossentropy")
    char metric_name[20];     // Name of the evaluation metric (e.g., "accuracy")
} ModelConfig;

// Function declarations

/* 
 * Function to create a new CNN model, only the constructor is loaded.
 *
 * This function initializes a new CNN model with the provided input and output dimensions.
 * The model is allocated in memory, and its parameters are initialized.
 * The actual input data will be provided separately.
 *
 * Parameters:
 * - input_width: Width of the input data.
 * - input_height: Height of the input data.
 * - input_channels: Number of channels in the input data.
 * - output_width: Width of the output data.
 * - output_height: Height of the output data.
 * - output_channels: Number of channels in the output data.
 *
 * Returns:
 * - A pointer to the newly created Model struct if successful, NULL otherwise.
 *
 * Usage example:
 * 
 * Model* model = create(28, 28, 1, 10, 10, 1);
 * if (model == NULL) {
 *     // Handle error
 * }
 */
Model* create(int input_width, int input_height, int input_channels,
              int output_width, int output_height, int output_channels);

/**
 * Adds a convolutional layer to the model.
 *
 * @param model A pointer to the Model struct.
 * @param filters The number of filters in the convolutional layer.
 * @param kernel_size The size of the convolutional kernel.
 * @param stride The stride value for the convolution operation.
 * @param padding The padding value for the convolution operation.
 * @param activation The activation function to be used in the convolutional layer.
 */
void add_convolutional_layer(Model* model, int filters, int kernel_size, int stride, int padding, char* activation);

/**
 * Adds a max pooling layer to the model.
 *
 * @param model A pointer to the Model struct.
 * @param pool_size The size of the pooling window.
 * @param stride The stride value for the pooling operation.
 */
void add_max_pooling_layer(Model* model, int pool_size, int stride);

/**
 * Adds a fully-connected layer to the model.
 *
 * @param model A pointer to the Model struct.
 * @param num_neurons The number of neurons in the fully-connected layer.
 * @param activation The activation function to be used in the fully-connected layer.
 */
void add_fully_connected_layer(Model* model, int num_neurons, char* activation);

/**
 * Adds a dropout layer to the model.
 *
 * @param model A pointer to the Model struct.
 * @param dropout_rate The dropout rate for the dropout layer.
 */
void add_dropout_layer(Model* model, float dropout_rate);

/**
 * Adds a flatten layer to the model.
 *
 * @param model A pointer to the Model struct.
 */
void add_flatten_layer(Model* model);

/**
 * Adds a softmax layer to the model.
 *
 * @param model A pointer to the Model struct.
 */
void add_softmax_layer(Model* model);

/**
 * Adds a ReLU layer to the model.
 *
 * @param model A pointer to the Model struct.
 */
void add_relu_layer(Model* model);

/**
 * Adds a sigmoid layer to the model.
 *
 * @param model A pointer to the Model struct.
 */
void add_sigmoid_layer(Model* model);

/**
 * Adds a tanh layer to the model.
 *
 * @param model A pointer to the Model struct.
 */
void add_tanh_layer(Model* model);


/* 
 * Function to compile the model with provided configuration.
 *
 * This function assigns the configuration settings provided in the ModelConfig struct
 * to the respective fields in the Model struct, effectively compiling the model.
 *
 * Parameters:
 * - model: Pointer to the CNN model to be compiled.
 * - config: Configuration settings for the model.
 *           Config includes the optimizer name, learning rate, loss function, batch size, number of epochs, and evaluation metric.
 *
 * Usage example:
 * 
 * ModelConfig config = { "Adam", 0.001f, "categorical_crossentropy", 64, 10, "accuracy" };
 * compile_model(model, config);
 */
void compile(Model* model, ModelConfig config);

// Function to train the CNN model
void train(Model* model, InputData** training_data, int num_samples, int num_epochs);

// Function declaration for the test function
float test(Model* model, InputData** test_data, int num_samples);

/* 
 * Function to free memory allocated for the CNN model.
 *
 * This function releases all dynamically allocated memory associated with the model,
 * including input data, layers, and the model itself.
 *
 * Parameters:
 * - model: Pointer to the CNN model to be freed.
 *
 * Usage example:
 * 
 * free_model(model);
 */
void free_model(Model* model);

#endif /* CNN_H */
