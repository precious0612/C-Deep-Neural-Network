/* model/model.h
 *
 * This file defines the core of the Convolutional Neural Network (CNN) model, serving as the brain
 * that orchestrates the entire learning process. It encapsulates the model's configuration, layers,
 * and the algorithms required for training and inference.
 *
 * The Model structure acts as a container, holding the input and output dimensions, optimizer settings,
 * loss function, evaluation metric, and a collection of layers that make up the neural network architecture.
 * This file provides a high-level interface for interacting with the model, allowing you to create, compile,
 * train, and evaluate the CNN model with ease.
 *
 * Key functionalities include:
 *
 * 1. Constructing a new model by specifying the input and output dimensions.
 * 2. Adding layers to the model, such as convolutional, pooling, fully connected, dropout, and activation layers.
 * 3. Compiling the model by configuring the optimizer, loss function, and evaluation metric.
 * 4. Performing forward and backward passes through the model during training and inference.
 * 5. Training the model on a dataset, monitoring the loss and accuracy during the training process.
 * 6. Evaluating the model's performance on a separate dataset.
 * 7. Updating the model's weights and resetting gradients during the training process.
 * 8. Printing detailed information about the model's architecture and configuration.
 * 9. Properly deallocating memory used by the model when it is no longer needed.
 *
 * This header file serves as the entry point for building and utilizing the CNN model, providing a
 * consistent and intuitive interface for developers to harness the power of deep learning in their
 * applications.
 */

#ifndef MODEL_H
#define MODEL_H

#include "../dataset.h"
#include "layer/layer.h"
#include "../optimizer/optimizer.h"
#include "../utils/utils.h"

// Define structure for holding model
typedef struct {
    Dimensions input;      // Input dimensions
    Dimensions output;     // Output dimensions
    char optimizer_name[20];       // Name of the optimizer (e.g., "SGD", "Adam")
    float learning_rate;      // Learning rate for optimization
    Optimizer* optimizer;  // Pointer to the optimizer
    char loss_function[30];   // Name of the loss function (e.g., "categorical_crossentropy")
    LossFunction loss_fn;   // Pointer to the loss function
    char metric_name[20];     // Name of the evaluation metric (e.g., "accuracy")
    Layer** layers;           // Pointer to the first layer pointer in the model
    int num_layers;           // Number of layers in the model
} Model;

// Function declarations

/*
 * Creates a new model with the specified input and output dimensions.
 *
 * Parameters:
 * - input: The dimensions of the input data (width, height, and channels).
 * - output: The dimensions of the output data (width, height, and channels).
 *
 * Returns:
 * - A pointer to the newly created Model struct, or NULL if memory allocation fails.
 *
 * Usage example:
 *
 * Dimensions input_dim = {32, 32, 3};
 * Dimensions output_dim = {1, 1, 10};
 * Model* model = create_model(input_dim, output_dim);
 * if (model == NULL) {
 *     // Handle error
 * }
 */
Model* create_model(Dimensions input, Dimensions output);

/*
 * Adds a layer to the model.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * - layer: A pointer to the Layer struct to be added.
 *
 * Usage example:
 *
 * Layer* conv_layer = create_layer(CONVOLUTIONAL, conv_params);
 * add_layer_to_model(model, conv_layer);
 */
void add_layer_to_model(Model* model, Layer* layer);

/*
 * Adds a layer to the model with specified parameters.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * - layer_type: A string representing the type of layer (e.g., "convolutional", "max_pooling", "fully_connected").
 * - num_filters: The number of filters for the layer (only applicable to convolutional and fully connected layers).
 * - filter_size: The size of the filter for the layer (only applicable to convolutional and pooling layers).
 * - stride: The stride value for the layer (only applicable to convolutional and pooling layers).
 * - padding: The padding value for the layer (only applicable to convolutional layers).
 * - activation: A string representing the activation function for the layer (e.g., "relu", "sigmoid", "tanh").
 * - dropout_rate: The dropout rate for the layer (only applicable to dropout layers).
 *
 * Usage example:
 *
 * add_layer(model, "convolutional", 32, 3, 1, 1, "relu", 0.0);
 * add_layer(model, "max_pooling", 0, 2, 2, 0, "", 0.0);
 * add_layer(model, "fully_connected", 64, 0, 0, 0, "relu", 0.0);
 * add_layer(model, "dropout", 0, 0, 0, 0, "", 0.5);
 */
void add_layer(Model* model, char* layer_type, int num_filters, int filter_size, int stride, int padding, char* activation, float dropout_rate);

/*
 * Compiles the model with the specified optimizer, loss function, and evaluation metric.
 *
 * Parameters:
 * - model: A pointer to the Model struct.
 * - optimizer: A string representing the optimizer to be used (e.g., "sgd", "adam", "rmsprop").
 * - learning_rate: The learning rate for the optimizer.
 * - loss_function: A string representing the loss function to be used (e.g., "categorical_crossentropy", "mean_squared_error").
 * - metric_name: A string representing the evaluation metric to be used (e.g., "accuracy").
 *
 * Usage example:
 *
 * compile_model(model, "adam", 0.001, "categorical_crossentropy", "accuracy");
 */
void compile_model(Model* model, char* optimizer, float learning_rate, char* loss_function, char* metric_name);

/*
 * Prints detailed information about the model's architecture, including the input and output dimensions,
 * optimizer settings, loss function, evaluation metric, and the layers comprising the neural network.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 *
 * Usage example:
 *
 * print_model_info(model);
 */
void print_model_info(Model* model);

/*
 * Performs the forward propagation of input data through the model's layers, computing
 * the output at each stage of the neural network.
 * 
 * This function orchestrates the forward pass through the model, invoking the forward
 * propagation routine for each layer sequentially, starting from the input layer and
 * progressing towards the output layer. It enables the model to generate predictions
 * or output representations based on the provided input data.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 * - input: A 3D array containing the input data to be propagated through the model.
 *
 * Returns:
 * - A 3D array containing the output data computed by the model's final layer.
 *
 * Usage example:
 * 
 * float*** input = create_3d_array(32, 32, 3);
 * float*** output = forward_pass(model, input);
 * if (output == NULL) {
 *    // Handle error
 * }
 */
float*** forward_pass(Model* model, float*** input);

/*
 * Performs the backward propagation of gradients through the model's layers, updating
 * the weights and biases based on the computed gradients.
 * 
 * This function orchestrates the backward pass through the model, invoking the backward
 * propagation routine for each layer in reverse order, starting from the output layer and
 * propagating the gradients towards the input layer. It enables the model to update its
 * internal parameters (weights and biases) based on the computed gradients, facilitating
 * the learning process during training.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 * - input: A 3D array containing the input data used during the forward pass.
 * - output_grad: A 3D array containing the gradients of the output with respect to the loss.
 *
 * Usage example:
 * 
 * float*** input = create_3d_array(32, 32, 3);
 * float*** output_grad = create_3d_array(1, 1, 10);
 * backward_pass(model, input, output_grad);
 */
void backward_pass(Model* model, float*** input, float*** output_grad);

/*
 * Performs the forward propagation of a batch of input data through the model's layers,
 * computing the output for each sample in the batch.
 * 
 * This function enables efficient processing of multiple input samples simultaneously by invoking
 * the forward pass routine for each sample in the batch. It is particularly useful during training,
 * where batches of data are typically processed together for improved performance and convergence.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 * - batch_inputs: A 4D array containing the batch of input data to be propagated through the model.
 * - batch_size: The number of samples in the batch.
 *
 * Returns:
 * - A 4D array containing the output data computed by the model's final layer for each sample in the batch.
 *
 * Usage example:
 * 
 * float**** batch_inputs = create_4d_array(32, 32, 3, 64);
 * float**** batch_outputs = forward_pass_batch(model, batch_inputs, 64);
 * if (batch_outputs == NULL) {
 *   // Handle error
 * }
 */
float**** forward_pass_batch(Model* model, float**** batch_inputs, int batch_size);

/*
 * Performs the backward propagation of gradients through the model's layers for a batch
 * of input data, updating the weights and biases based on the computed gradients.
 * 
 * This function orchestrates the backward pass through the model for a batch of input data,
 * invoking the backward propagation routine for each sample in the batch. It enables the model
 * to update its internal parameters (weights and biases) based on the computed gradients,
 * facilitating the learning process during training with batched data.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 * - batch_inputs: A 4D array containing the batch of input data used during the forward pass.
 * - batch_output_grads: A 4D array containing the gradients of the output with respect to the loss for each sample in the batch.
 * - batch_size: The number of samples in the batch.
 *
 * Usage example:
 * 
 * float**** batch_inputs = create_4d_array(32, 32, 3, 64);
 * float**** batch_output_grads = create_4d_array(1, 1, 10, 64);
 * backward_pass_batch(model, batch_inputs, batch_output_grads, 64);
 * update_model_weights(model);
 * reset_model_grads(model);
 */
void backward_pass_batch(Model* model, float**** batch_inputs, float**** batch_output_grads, int batch_size);

/*
 * Updates the weights and biases of the model's layers based on the computed gradients
 * and the specified optimization algorithm.
 * 
 * This function invokes the weight update routine for each layer in the model, applying
 * the specified optimization algorithm (e.g., Stochastic Gradient Descent, Adam, RMSprop)
 * to adjust the weights and biases based on the computed gradients. It is typically called
 * after performing the backward pass during training, enabling the model to learn from the
 * data and improve its performance iteratively.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 *
 * Usage example:
 * 
 * update_model_weights(model);
 * reset_model_grads(model);
 */
void update_model_weights(Model* model);

/*
 * Resets the gradients for all layers in the model to zero, preparing for the next
 * iteration of the training process.
 * 
 * This function is typically called before starting a new iteration of the training process,
 * resetting the gradients accumulated during the previous iteration. It ensures that the
 * gradients are properly initialized before performing the forward and backward passes,
 * preventing the accumulation of gradients from previous iterations and enabling a fresh
 * start for the learning process.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 *
 * Usage example:
 * 
 * reset_model_grads(model);
 */
void reset_model_grads(Model* model);

/*
 * Trains the CNN model on the provided dataset for a specified number of epochs,
 * monitoring the loss and accuracy during the training process.
 * 
 * This function orchestrates the entire training process for the CNN model. It iterates
 * over the specified number of epochs, dividing the dataset into batches for efficient
 * processing. During each epoch, it performs the forward and backward passes, updates
 * the model's weights, and computes the loss and accuracy for the current batch. Progress
 * is printed to the console, and if a validation dataset is provided, the model's performance
 * on the validation data is evaluated and reported after each epoch.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 * - dataset: A pointer to the Dataset struct containing the training data.
 * - num_epochs: The number of epochs (complete passes through the dataset) to train the model for.
 *
 * Usage example:
 * 
 * train_model(model, train_dataset, 10);
 */
void train_model(Model* model, Dataset* dataset, int num_epochs);

/*
 * Evaluates the performance of the CNN model on the provided dataset, computing
 * the accuracy or other specified metric.
 * 
 *  This function is typically used to assess the model's generalization performance on
 * a separate test or validation dataset after training. It performs a forward pass through
 * the model for each sample in the dataset, computes the predicted output, and compares it
 * to the ground truth labels to calculate the evaluation metric. The final result is returned
 * as a floating-point value, allowing further analysis or reporting of the model's performance.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 * - dataset: A pointer to the Dataset struct containing the evaluation data.
 *
 * Returns:
 * - A floating-point value representing the model's performance on the provided dataset,
 *   according to the specified evaluation metric (e.g., accuracy, precision, recall).
 *
 * Usage example:
 * 
 * float accuracy = evaluate_model(model, test_dataset);
 * printf("Model accuracy on test dataset: %.2f\n", accuracy);
 */
float evaluate_model(Model* model, Dataset* dataset);

/*
 * Saves the weights and biases of the CNN model to a file in HDF5 format.
 *
 * This function saves the weights and biases of all layers in the CNN model to a file
 * in the HDF5 format, which is a widely used format for storing and transferring scientific
 * data. The weights and biases are organized in groups, with each group representing a layer
 * in the model. The function can be used to save the model's state during or after training,
 * allowing for later resumption or deployment of the trained model.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 * - filename: A string containing the name of the file to save the model weights and biases to.
 *
 * Usage example:
 *
 * save_model_weights(model, "model_weights.h5");
 */
void save_model_weights(Model *model, const char *filename);

/*
 * Loads the weights and biases of the CNN model from a file in HDF5 format.
 *
 * This function loads the weights and biases of all layers in the CNN model from a file
 * in the HDF5 format. The function assumes that the file was previously created using
 * the `save_model_weights` function, and that the model architecture matches the one used
 * when saving the weights and biases. The loaded weights and biases are assigned to the
 * corresponding layers in the provided model, allowing for the restoration of a previously
 * trained model or the initialization of a new model with pre-trained weights.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model.
 * - filename: A string containing the name of the file to load the model weights and biases from.
 *
 * Usage example:
 *
 * load_model_weights(model, "model_weights.h5");
 */
void load_model_weights(Model *model, const char *filename);

void load_vgg16_weights(Model *model, const char *filename);

/*
 * Frees the memory allocated for the CNN model, including its layers, weights,
 * and associated data structures.
 * 
 * This function is called when the CNN model is no longer needed, ensuring that
 * all dynamically allocated memory associated with the model is properly released
 * and returned to the system. It is essential to call this function to prevent
 * memory leaks and maintain good memory management practices in your application.
 *
 * Parameters:
 * - model: A pointer to the Model struct representing the CNN model to be deallocated.
 *
 * Usage example:
 * 
 * delete_model(model);
 */
void delete_model(Model* model);

#endif /* MODEL_H */
