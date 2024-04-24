/* model/model.h 
 * 
 * This file it defines the interface for the model object in the CNN. 
 * The model object contains the necessary methods to perform 
 * the forward pass, backward pass, and update the parameters for the entire model.
 *
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

// Create a new model
Model* create_model(Dimensions input, Dimensions output);

// Add a layer to the model
void add_layer_to_model(Model* model, Layer* layer);

// Add a layer
void add_layer(Model* model, char* layer_type, int num_filters, int filter_size, int stride, int padding, char* activation, float dropout_rate);

// Compile the model
void compile_model(Model* model, char* optimizer, float learning_rate, char* loss_function, char* metric_name);

// Print model information
void print_model_info(Model* model);

// Perform forward pass through the model
void forward_pass(Model* model, float*** input, float*** output);

// Perform backward pass through the model
void backward_pass(Model* model, float*** input, float*** output, float*** output_grad);

// Perform forward pass through the model for a batch of inputs
void forward_pass_batch(Model* model, float**** batch_inputs, float**** batch_outputs, int batch_size);

// Perform backward pass through the model for a batch of inputs
void backward_pass_batch(Model* model, float**** batch_inputs, float**** batch_outputs, float**** batch_output_grads, float learning_rate, int batch_size);

// Update the model weights
void update_model_weights(Model* model);

// Reset the model gradients
void reset_model_grads(Model* model);

// Train the model on a dataset
void train_model(Model* model, Dataset* dataset, int num_epochs);

// Evaluate the model on a dataset
float evaluate_model(Model* model, Dataset* dataset);

// Free memory allocated for the model
void delete_model(Model* model);

#endif /* MODEL_H */
