/* CNN.h (high-level) */

#ifndef CNN_H
#define CNN_H

#include "model/model.h"  // Include model.h to access Model structure

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
 * Model* model = create_model(28, 28, 1, 10, 10, 1);
 * if (model == NULL) {
 *     // Handle error
 * }
 */
Model* create_model(int input_width, int input_height, int input_channels,
                    int output_width, int output_height, int output_channels);

/* 
 * Function to add a layer to the CNN model.
 *
 * This function adds a layer with specified type and parameters to the CNN model.
 * Layers are appended to the end of the model's layer list.
 *
 * Parameters:
 * - model: Pointer to the CNN model to which the layer will be added.
 * - type: Type of the layer to be added (CONVOLUTIONAL, POOLING, FULLY_CONNECTED, DROPOUT).
 * - params: Parameters specific to the layer type.
 *           For convolutional layers, this would include the number of filters, kernel size, stride, padding and activation function.
 *           For pooling layers, this would include the pool size, stride and pool type.
 *           For fully connected layers, this would include the number of neurons and activation function.
 *           For dropout layers, this would only include the dropout rate.
 *
 * Usage example:
 * 
 * LayerParams conv_params = { .conv_params = { 32, 3, 1, 1, "relu" } };
 * add_layer(model, CONVOLUTIONAL, conv_params);
 */
void add_layer(Model* model, LayerType type, LayerParams params);

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
void compile_model(Model* model, ModelConfig config);

// Function to print model information
void print_model_info(Model* model);

// Function to check if the final layer output shape matches the output information
int check_output_shape(Model *model);

// Function to train the CNN model
void train_model(Model* model, InputData** training_data, int num_samples, int num_epochs);

// Function declaration for the test function
float test_model(Model* model, InputData** test_data, int num_samples);

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
