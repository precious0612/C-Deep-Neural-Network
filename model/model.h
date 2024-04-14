/* model.h */

#ifndef MODEL_H
#define MODEL_H

#include "../input/dataset.h"
#include "layer/layer.h"

// Define structure for holding model configuration
typedef struct {
    // int input_shape[3];       // Shape of the input data (width, height, channels)
    // int output_shape[3];      // Shape of the output data (width, height, channels)
    char optimizer[20];       // Name of the optimizer (e.g., "SGD", "Adam")
    float learning_rate;      // Learning rate for optimization
    char loss_function[30];   // Name of the loss function (e.g., "categorical_crossentropy")
    char metric_name[20];     // Name of the evaluation metric (e.g., "accuracy")
} ModelConfig;

// Define structure for holding model
typedef struct {
    Dimensions input;      // Input dimensions
    Dimensions output;     // Output dimensions
    char optimizer[20];       // Name of the optimizer (e.g., "SGD", "Adam")
    float learning_rate;      // Learning rate for optimization
    char loss_function[30];   // Name of the loss function (e.g., "categorical_crossentropy")
    char metric_name[20];     // Name of the evaluation metric (e.g., "accuracy")
    Layer* layers;            // Pointer to the first layer in the model
} Model;

// Function declarations

// Function declaration for forward pass
void forward_pass(Model* model, InputData* input_data);

// Function declaration for backward pass
void backward_pass(Model* model, InputData* input_data);

// Function to perform a single training epoch
void train_epoch(Model* model, Dataset* training_dataset);

// Function declaration for evaluation
float evaluate_model(Model* model, Dataset* validation_dataset);

#endif /* MODEL_H */
