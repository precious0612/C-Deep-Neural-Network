//
//  model.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/24/24.
//

#ifndef model_h
#define model_h

#include "../dataset.h"
#include "layer/layer.h"
#include "optimizer/optimizer.h"
#include "loss/losses.h"
#include "metric/metric.h"

// MARK: - Define the params for optimizers
#define ADAM_BETA1      0.9f
#define ADAM_BETA2      0.999f
#define ADAM_EPSILON    1e-8f
#define RMSPROP_RHO     0.9f
#define RMSPROP_EPSILON 1e-8f

// MARK: Define the * interval length
#define MAX_BATCH_PRINT 100
#define PRINT_INTERVAL(total_batches) ((total_batches) <= MAX_BATCH_PRINT ? 1 : MAX_BATCH_PRINT)

#define MAX_BATCH_PROGRESS_STR_LEN 32
#define MAX_ERROR_MSG_LEN          64

#define MAX_DATASET_NAME_LEN 32
#define MAX_GROUP_NAME_LEN   64

// MARK: - Define the structure of model

typedef struct {
    Dimensions    input;
    Dimensions    output;
    Optimizer*    optimizer;
    LearningRate  learning_rate;
    LossFunction* loss;
    Layer**       layers;
    Metric        metric;
    int           num_layers;
} Model;

// MARK: Define the variables

typedef float*** Input;
typedef float*** Output;
typedef float*** InputGrad;
typedef float*** OutputGrad;

typedef int       BatchSize;
typedef float**** BatchedInputs;
typedef float**** BatchedOutputs;
typedef float**** BatchedOutputGrads;

typedef float Accuracy;
typedef float LossValue;
typedef int   Label;

// MARK: - Method Declarations

/// Creates a new model with the specified input and output dimensions.
/// - Parameters:
///   - input:  The dimensions of the input data (`width`, `height`, and `channels`).
///   - output: The dimensions of the output data (`width`, `height`, and `channels`).
/// - Returns: A pointer to the newly created `Model` struct, or `NULL` if memory allocation fails.
///
/// - Example Usage:
///     ```c
///     Dimensions input_dim  = {32, 32, 3};
///     Dimensions output_dim = {1, 1, 10};
///     Model* model = create_model(input_dim, output_dim);
///     ```
///
Model* create_model(Dimensions input, Dimensions output);

/// Adds a layer to the model.
/// - Parameters:
///   - model:        A pointer to the `Model` struct which the layer will be added into.
///   - layer_type:   The `LayerType` of the layer will be added into model.
///   - num_filters:  The number of kernels for the convolutional layer or the number of neurous for fullly connected layer.
///   - filter_size:  The size of kernel for the convolutional layer or the size of pool window for the pooling layer.
///   - stride:       The stride value for the layer (only applicable to convolutional and pooling layers).
///   - padding:      The padding value for the layer (only applicable to convolutional layers).
///   - activation:   The activation function for the layer (e.g., `RELU`, `SIGMOID`, `TANH`).
///   - pool_type:    The `PoolType` for the pooling layer. (e.g., `MAX` or `AVARAGE`)
///   - dropout_rate: The dropout rate for the layer (only applicable to dropout layers).
///
/// - Example Usage:
///     ```c
///     add_layer(model, CONVOLUTIONAL,   32, 3, 1, 1, RELU,    0,   0.0f);
///     add_layer(model, POOLING,         0,  2, 2, 0, 0,       MAX, 0.0f);
///     add_layer(model, FULLY_CONNECTED, 10, 0, 0, 0, SOFTMAX, 0,   0.0f);
///     ```
///
void add_layer(Model* model, LayerType layer_type, int num_filters, int filter_size, int stride, int padding, ActivationType activation, PoolType pool_type, float dropout_rate);

/// Compiles the model with the specified optimizer, loss function, and evaluation metric.
/// - Parameters:
///   - model: A pointer to the `Model` struct.
///   - optimizer_type: The optimizer type to be used (e.g., `SGD`, `ADAM`, `RMSPROP`).
///   - learning_rate: The learning rate for the optimizer.
///   - loss_type: The loss function to be used (e.g., `CrossEntropy`, `MSE`).
///   - metric: The evaluation metric to be used (e.g., `LOSS`, `ACCURACY`).
///
/// - Example Usage:
///     ```c
///     compile_model(model, "adam", 0.001, "categorical_crossentropy", "accuracy");
///     ```
///
void compile_model(Model* model, OptimizerType optimizer_type, LearningRate learning_rate, LossType loss_type, Metric metric);

/// Prints detailed information about the model's architecture, including the `input` and `output dimensions`, `optimizer settings`, `loss function`, `evaluation metric`, and `the layers `comprising the neural network.
/// - Parameter model: A pointer to the `Model` struct representing the model.
///
/// - Example Usage:
///     ```c
///     print_model_info(model);
///     ```
///
void print_model_info(const Model* model);

/// This function orchestrates the entire training process for the model.
/// It iterates over the specified number of epochs, dividing the dataset into batches for efficient processing. During each epoch, it performs the forward and backward passes, updates the model's weights, and computes the loss and accuracy for the current batch.
/// Progress is printed to the console, and if a validation dataset is provided, the model's performance on the validation data is evaluated and reported after each epoch.
/// - Parameters:
///   - model: A pointer to the `Model` struct representing the model.
///   - dataset: A pointer to the `Dataset` struct containing the training data.
///   - num_epochs: The number of epochs (complete passes through the dataset) to train the model for.
///   
/// - Example Usage:
///     ```c
///     train_model(model, train_dataset, 10);
///     ```
void train_model(Model* model, Dataset* dataset, int num_epochs);

/// Performs the forward propagation of input data through the model's layers, computing the output at each stage of the neural network.
/// - Parameters:
///   - model: A pointer to the `Model` struct representing the model.
///   - input: An `Input` containing the input data to be propagated through the model.
/// - Returns: The `Output` containing the output data computed by the model's final layer.
///
/// - Example Usage:
///     ```c
///     Input  input  = create_3d_array(32, 32, 3);
///     Output output = forward_pass(model, input);
///     ```
///
Output forward_pass(Model* model, Input input);

/// This function is typically used to assess the model's generalization performance on a separate test or validation dataset after training.
/// It performs a forward pass through the model for each sample in the dataset, computes the predicted output, and compares it to the ground truth labels to calculate the evaluation metric.
/// - Parameters:
///   - model: A pointer to the `Model` struct representing the model.
///   - dataset: A pointer to the `Dataset` struct containing the evaluation data.
/// - Returns: A floating-point value representing the model's performance on the provided dataset.
///
/// - Example Usage:
///     ```c
///     Accuracy accuracy = evaluate_model(model, test_dataset);
///     ```
///
Accuracy evaluate_model(Model* model, Dataset* dataset);

/// This function is called when the model is no longer needed, ensuring that all dynamically allocated memory associated with the model is properly released and returned to the system.
/// - Parameter model: A pointer to the `Model` struct representing the model to be deallocated.
/// 
/// - Example Usage:
///     ```c
///     delete_model(model);
///     ```
///
void delete_model(Model* model);

/// This function saves the weights and biases of all layers in the model to a file in the `HDF5` format, which is a widely used format for storing and transferring scientific data.
/// The weights and biases are organized in groups, with each group representing a layer in the model. The function can be used to save the model's state during or after training, allowing for later resumption or deployment of the trained model.
/// - Parameters:
///   - model: A pointer to the `Model` struct representing the model.
///   - filename: A string containing the name of the file to save the model weights and biases to.
///
/// - Example Usage:
///     ```c
///     save_model_weights(model, "model_weights.h5");
///     ```
///
void save_model_weights(Model* model, const char* filename);

/// This function loads the weights and biases of all layers in the CNN model from a file in the `HDF5` format.
/// The function assumes that the file was previously created using the `save_model_weights` function, and that the model architecture matches the one used when saving the weights and biases.
/// The loaded weights and biases are assigned to the corresponding layers in the provided model, allowing for the restoration of a previously trained model or the initialization of a new model with pre-trained weights.
/// - Parameters:
///   - model: A pointer to the `Model` struct representing the model.
///   - filename: A string containing the name of the file to load the model weights and biases from.
///   
/// - Example Usage:
///     ```c
///     load_model_weights(model, "model_weights.h5");
///     ```
///
void load_model_weights(Model* model, const char* filename);

/// This function loads the weights and biases of vgg16 `HDF5` format.
/// - Parameters:
///   - model: A pointer to the `Model` struct representing the model.
///   - filename: A string containing the name of the vgg16 file to load the model weights and biases from.
///
/// - Example Usage:
///     ```c
///     load_vgg16_weights(model, "vgg16.h5");
///     ```
///
void load_vgg16_weights(Model* model, const char* filename);

#endif /* model_h */
