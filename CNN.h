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
 * - output_width: Width of the output data. (normally 1)
 * - output_height: Height of the output data. (normally 1)
 * - output_channels: Number of channels in the output data. (e.g., number of classes)
 *
 * Returns:
 * - A pointer to the newly created Model struct if successful, NULL otherwise.
 *
 * Usage example:
 * 
 * Model* model = create(28, 28, 3, 1, 1, 10);
 * if (model == NULL) {
 *     // Handle error
 * }
 */
Model* create(int input_width, int input_height, int input_channels,
              int output_width, int output_height, int output_channels);

/*
 * Creates and compiles a CNN model from a JSON file.
 *
 * This function reads the model configuration from a JSON file and creates a CNN model based on that configuration.
 * The JSON file just like this:
 * {
 *     "input_shape": [28, 28, 1],
 *     "output_shape": [10],
 *     "optimizer": "Adam",
 *     "learning_rate": 0.001,
 *     "loss_function": "categorical_crossentropy",
 *     "batch_size": 32,
 *     "num_epochs": 10,
 *     "metric_name": "accuracy",
 *     "layers": [
 *         {
 *             "type": "convolutional",
 *             "params": {
 *                 "num_filters": 32,
 *                 "filter_size": 3,
 *                 "stride": 1,
 *                 "padding": "same",
 *                 "activation": "relu"
 *             }
 *         },
 *         {
 *             "type": "pooling",
 *             "params": {
 *                 "pool_size": 2,
 *                 "stride": 2,
 *                 "pool_type": "max"
 *             }
 *         },
 *         {
 *             "type": "fully_connected",
 *             "params": {
 *                 "num_neurons": 128,
 *                 "activation": "relu"
 *             }
 *         }
 *     ]
 * }
 * ( model_config.json )
 * 
 * Parameters:
 * - filename: The path to the JSON file containing the model configuration.
 * 
 * Returns:
 * - A pointer to the compiled Model struct, or NULL if the file could not be loaded or the model could not be created.
 * 
 * Usage example:
 * Model* model = create_model_from_json("model_config.json");
 * if (model == NULL) {
 *     // Handle error
 * }
 */
Model* create_model_from_json(const char* filename);

/*
 * Saves the model configuration and architecture to a JSON file.
 *
 * This function saves the model configuration and architecture to a JSON file.
 *
 * Parameters:
 * - model: A pointer to the Model struct to be saved.
 * - filename: The path to the file where the model will be saved.
 * 
 * Returns:
 * - 0 if the model was saved successfully, a non-zero value otherwise.
 * 
 * Usage example:
 * Model* model = create(...); // Create your model
 * // ... Add layers and compile the model
 *
 * int result = save_model_to_json(model, "model_config.json");
 * if (result != 0) {
 *     // Handle error
 * }
 */
int save_model_to_json(Model* model, const char* filename);

/*
 * Adds a convolutional layer to the model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * - filters: The number of filters in the convolutional layer.
 * - kernel_size: The size of the convolutional kernel.
 * - stride: The stride value for the convolution operation.
 * - padding: The padding value for the convolution operation.
 * - activation: The activation function to be used in the convolutional layer.
 * 
 * Usage example:
 * 
 * add_convolutional_layer(model, 32, 3, 1, 1, "relu");
 * 
 */
void add_convolutional_layer(Model* model, int filters, int kernel_size, int stride, int padding, char* activation);

/*
 * Adds a max pooling layer to the model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * - pool_size: The size of the pooling window.
 * - stride: The stride value for the pooling operation.
 * 
 * Usage example:
 * 
 * add_max_pooling_layer(model, 2, 2);
 * 
 */
void add_max_pooling_layer(Model* model, int pool_size, int stride);

/*
 * Adds a fully-connected layer to the model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * - num_neurons: The number of neurons in the fully-connected layer.
 * - activation: The activation function to be used in the fully-connected layer.
 * 
 * Usage example:
 * 
 * add_fully_connected_layer(model, 128, "relu");
 * 
 */
void add_fully_connected_layer(Model* model, int num_neurons, char* activation);

/*
 * Adds a dropout layer to the model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * - dropout_rate: The dropout rate for the dropout layer.
 * 
 * Usage example:
 * 
 * add_dropout_layer(model, 0.5);
 * 
 */
void add_dropout_layer(Model* model, float dropout_rate);

/*
 * Adds a flatten layer to the model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * 
 * Usage example:
 * 
 * add_flatten_layer(model);
 * 
 */
void add_flatten_layer(Model* model);

/*
 * Adds a softmax layer to the model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * 
 * Usage example:
 * 
 * add_softmax_layer(model);
 * 
 */
void add_softmax_layer(Model* model);

/*
 * Adds a ReLU layer to the model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * 
 * Usage example:
 * 
 * add_relu_layer(model);
 */
void add_relu_layer(Model* model);

/*
 * Adds a sigmoid layer to the model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * 
 * Usage example:
 * 
 * add_sigmoid_layer(model);
 * 
 */
void add_sigmoid_layer(Model* model);

/*
 * Adds a tanh layer to the model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * 
 * Usage example:
 * 
 * add_tanh_layer(model);
 */
void add_tanh_layer(Model* model);


/* 
 * Function to compile the model with provided configuration.
 *
 * This function assigns the configuration settings provided in the ModelConfig struct
 * to the respective fields in the Model struct, effectively compiling the model.
 *
 * - model: Pointer to the CNN model to be compiled.
 * - config: Configuration settings for the model.
 *           Config includes the optimizer name, learning rate, loss function, batch size, number of epochs, and evaluation metric.
 *
 * Usage example:
 * 
 * ModelConfig config = { "Adam", 0.001f, "categorical_crossentropy", "accuracy" };
 * compile_model(model, config);
 */
void compile(Model* model, ModelConfig config);

/*
 * Trains the model on the provided dataset.
 *
 * This function trains the model on the provided dataset for the specified number of epochs.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * - dataset: A pointer to the Dataset struct containing the training data.
 * - epochs: The number of epochs to train the model.
 * Tips: Dataset struct has been defined in dataset.h
 * 
 * Usage example:
 * 
 * train(model, dataset, 10);
 */
void train(Model* model, Dataset* dataset, int epochs);

/*
 * Evaluates the model on the provided dataset.
 *
 * This function evaluates the model on the provided dataset and returns the evaluation metric value.
 *
 * Parameters:
 * - model A pointer to the Model struct.
 * - dataset A pointer to the Dataset struct containing the evaluation data.
 * 
 * Returns:
 * - The evaluation metric value (e.g., accuracy) for the model on the given dataset.
 * 
 * Usage example:
 * 
 * printf("The accuracy is: %f", evaluate(model, dataset));
 */
float evaluate(Model* model, Dataset* dataset);

/*
 * Predicts the output of the model for a given input.
 *
 * This function takes the input data and predicts the output using the trained model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * - input: A 3D array representing the input data.
 * 
 * Returns:
 * - A pointer to a 3D array to store the predicted output.
 * 
 * Usage example:
 * 
 * float*** input = ...;  // Load input data
 * float*** output = predict(model, input);
 * // Use the predicted output
 * 
 */
float*** predict(Model* model, float*** input);

/*
 * Loads a pre-trained model from a file.
 *
 * Parameters:
 * - filename: The path to the file containing the pre-trained model.
 * 
 * Returns:
 * - A pointer to the loaded Model struct, or NULL if the file could not be loaded.
 * 
 * Usage example:
 * 
 * Model* model = load_model("model.json");
 */
Model* load_model(const char* filename);

/*
 * Saves the current model to a file.
 *
 * Parameters:
 * - model: A pointer to the Model struct to be saved.
 * - filename: The path to the file where the model will be saved.
 * 
 * Returns:
 * - 0 if the model was saved successfully, a non-zero value otherwise.
 * 
 * Usage example:
 * 
 * save_model(model, "model.json");
 */
int save_model(Model* model, const char* filename);

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
