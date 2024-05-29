//
//  cdnn.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/28/24.
//

#ifndef cdnn_h
#define cdnn_h

#include "model/model.h"

// MARK: - Define the config for model creators

typedef struct {
    char optimizer[20];       // Name of the optimizer (e.g., "SGD", "Adam")
    float learning_rate;      // Learning rate for optimization
    char loss_function[30];   // Name of the loss function (e.g., "categorical_crossentropy")
    char metric_name[20];     // Name of the evaluation metric (e.g., "accuracy")
} ModelConfig;

// MARK: - APIs

/// This function initializes a new `Model` with the provided input and output dimensions.
/// - Parameters:
///   - input_width:     Width of the input data.
///   - input_height:    Height of the input data.
///   - input_channels:  Number of channels in the input data.
///   - output_width:    Width of the output data. (normally 1)
///   - output_height:   Height of the output data. (normally 1)
///   - output_channels: Number of channels in the output data. (e.g., number of classes)
/// - Returns: A pointer to the newly created `Model` struct if successful, `NULL` otherwise.
///
/// - Example Usage:
///     ```c
///     Model* model = create(28, 28, 3, 1, 1, 10);
///     ```
///
Model* create(int input_width, int input_height, int input_channels,
              int output_width, int output_height, int output_channels);

/// This function reads the model configuration from a JSON file and creates a `Model` based on that configuration.
/// - Parameter filename: The path to the JSON file containing the model configuration.
/// - Returns: A pointer to the compiled `Model` struct, or `NULL` if the file could not be loaded or the model could not be created.
///
/// - Example Usage:
///     ```c
///     Model* model = create_model_from_json("model_config.json");
///     ```
///
/** The JSON file just like this:
     ```json
 {
     "input_shape": [28, 28, 1],
     "output_shape": [10],
     "optimizer": "Adam",
     "learning_rate": 0.001,
     "loss_function": "categorical_crossentropy",
     "batch_size": 32,
     "num_epochs": 10,
     "metric_name": "accuracy",
     "layers": [
         {
             "type": "convolutional",
             "params": {
                 "num_filters": 32,
                 "filter_size": 3,
                 "stride": 1,
                 "padding": "same",
                 "activation": "relu"
             }
         },
         {
             "type": "pooling",
             "params": {
                 "pool_size": 2,
                 "stride": 2,
                 "pool_type": "max"
             }
         },
         {
             "type": "fully_connected",
             "params": {
                 "num_neurons": 128,
                 "activation": "relu"
             }
         }
     ]
 }
     ```
 ( model_config.json )
 */
///
Model* create_model_from_json(const char* filename);

/// Saves the model configuration and architecture to a JSON file.
/// - Parameters:
///   - model:    A pointer to the `Model` struct to be saved.
///   - filename: The path to the file where the model will be saved.
/// - Returns: 0 if the model was saved successfully, a non-zero value otherwise.
///
/// - Example Usage:
///     ```c
///     Model* model = create(...); // Create your model
///     // ... Add layers and compile the model
///
///     int result = save_model_to_json(model, "model_config.json");
///     ```
///
int save_model_to_json(Model* model, const char* filename);

/// Adds a `convolutiona`l layer to the model.
/// - Parameters:
///   - model:       A pointer to the `Model` struct.
///   - filters:     The number of filters in the convolutional layer.
///   - kernel_size: The size of the convolutional kernel.
///   - stride:      The stride value for the convolution operation.
///   - padding:     The padding value for the convolution operation.
///   - activation:  The activation function to be used in the convolutional layer.
///
/// - Example Usage:
///     ```c
///     add_convolutional_layer(model, 32, 3, 1, 1, "relu");
///     ```
///
void add_convolutional_layer(Model* model, int filters, int kernel_size, int stride, int padding, char* activation);

/// Adds a `pooling` layer to the model.
/// - Parameters:
///   - model:     A pointer to the `Model` struct.
///   - pool_size: The size of the pooling window.
///   - stride:    The stride value for the pooling operation.
///   - pool_type: The type of pooling layer.
///
/// - Example Usage:
///     ```c
///     add_pooling_layer(model, 2, 2, "max");
///     ```
///
void add_pooling_layer(Model* model, int pool_size, int stride, char* pool_type);

/// Adds a `fully-connected` layer to the model. (Allow after convolution layer without flatten)
/// - Parameters:
///   - model: A pointer to the `Model` struct.
///   - num_neurons: The number of neurons in the fully-connected layer.
///   - activation: The activation function to be used in the fully-connected layer.
///
/// - Example Usage:
///     ```c
///     add_fully_connected_layer(model, 128, "relu");
///     ```
///
void add_fully_connected_layer(Model* model, int num_neurons, char* activation);

/// Adds a `dropout` layer to the model.
/// - Parameters:
///   - model:        A pointer to the `Model` struct.
///   - dropout_rate: The dropout rate for the dropout layer.
///
/// - Example Usage:
///     ```c
///     add_dropout_layer(model, 0.5);
///     ```
///
void add_dropout_layer(Model* model, float dropout_rate);

/// Adds a `softmax` layer to the model.
/// - Parameter model: A pointer to the `Model` struct.
///
/// - Example Usage:
///     ```c
///     add_softmax_layer(model);
///     ```
///
void add_softmax_layer(Model* model);

/// Adds a `ReLU` layer to the model.
/// - Parameter model: A pointer to the `Model` struct.
///
/// - Example Usage:
///     ```c
///     add_relu_layer(model);
///     ```
///
void add_relu_layer(Model* model);

/// Adds a `Sigmoid` layer to the model.
/// - Parameter model: A pointer to the `Model` struct.
///
/// - Example Usage:
///     ```c
///     add_sigmoid_layer(model);
///     ```
///
void add_sigmoid_layer(Model* model);

/// Adds a `tanh` layer to the model.
/// - Parameter model: A pointer to the `Model` struct.
///
/// - Example Usage:
///     ```c
///     add_tanh_layer(model);
///     ```
///
void add_tanh_layer(Model* model);

/// This function assigns the configuration settings provided in the `ModelConfig` struct to the respective fields in the `Model` struct, effectively compiling the model.
/// - Parameters:
///   - model:  Pointer to the `Model` to be compiled.
///   - config: Configuration settings for the model. Config includes the `optimizer name`, `learning rate`, `loss function`, `batch size`, `number of epochs`, and `evaluation metric`.
///
/// - Example Usage:
///     ```c
///     ModelConfig config = { "Adam", 0.001f, "categorical_crossentropy", "accuracy" };
///     compile(model, config);
///     ```
///
void compile(Model* model, ModelConfig config);

/// This function saves the weights of all layers in the `Model` to a file in the `HDF5` format.
/// The weights are organized in groups, with each group representing a layer in the model. The function can be used to save the model's state during or after training, allowing for later resumption or deployment of the trained model.
/// - Parameters:
///   - model:    A pointer to the `Model` struct representing the model.
///   - filename: A string containing the name of the file to save the model weights to.
///
/// - Example Usage:
///     ```c
///     save_weights(model, "model_weights.h5");
///     ```
///
void save_weights(Model* model, const char* filename);

/// This function loads the weights of all layers in the `Model` from a file in the `HDF5` format. The function assumes that the file was previously created using the `save_weights` method of this file, and that the model architecture matches the one used when saving the weights.
/// The loaded weights are assigned to the corresponding layers in the provided model, allowing for the restoration of a previously trained model or the initialization of a new model with pre-trained weights.
/// - Parameters:
///   - model:    A pointer to the `Model` struct representing the model.
///   - filename: A string containing the name of the file to load the model weights from.
///
/// - Example Usage:
///     ```c
///     load_weights(model, "model_weights.h5");
///     ```
///
void load_weights(Model* model, const char* filename);

/// This function creates and compiles the `VGG16` CNN model architecture.
/// If the `load_pretrained` parameter is set to 1, and the `weights_file` is provided, the function will load the pre-trained weights from the specified file. Otherwise, the model will be initialized with random weights.
/// - Parameters:
///   - weights_file:    A string containing the path to the file with pre-trained weights. If `NULL`, the model will be initialized with random weights.
///   - load_pretrained: An integer flag (`0` or `1`) indicating whether to load pre-trained weights.
///   - num_classes:     The number of output classes for the model.
///   - config:          A `ModelConfig` struct containing the configuration parameters for the model.
/// - Returns: A pointer to the compiled `Model` struct representing the `VGG16` model.
///
/// - Example Usage:
///     ```c
///     ModelConfig config = {"Adam", 0.001, "categorical_crossentropy", "accuracy"};
///     Model* vgg16_model = load_vgg16("vgg16_weights.h5", 1, 1000, config);
///     ```
///
Model* load_vgg16(const char* weights_file, int load_pretrained, int num_classes, ModelConfig config);

/// This function trains the model on the provided dataset for the specified number of epochs.
/// - Parameters:
///   - model: A pointer to the `Model` struct.
///   - dataset: A pointer to the `Dataset` struct containing the training data.
///   - epochs: The number of epochs to train the model.
///
/// - Example Usage:
///     ```c
///     train(model, dataset, 10);
///     ```
///
void train(Model* model, Dataset* dataset, int epochs);

/// This function evaluates the model on the provided dataset and returns the evaluation metric value.
/// - Parameters:
///   - model:   A pointer to the `Model` struct.
///   - dataset: A pointer to the `Dataset` struct containing the evaluation data.
/// - Returns: The evaluation metric value (e.g., accuracy) for the model on the given dataset.
///
/// - Example Usage:
///     ```c
///     printf("The accuracy is: %f", evaluate(model, dataset));
///     ```
///
float evaluate(Model* model, Dataset* dataset);

/// This function takes the input data and predicts the output using the trained model.
/// - Parameters:
///   - model: A pointer to the `Model` struct.
///   - input: A 3D array representing the input data.
/// - Returns: A pointer to a 3D array to store the predicted output.
///
/// - Example Usage:
///     ```c
///     Input  input  = ...;  // Load input data
///     Output output = predict(model, input);
///     ```
///
Output predict(Model* model, Input input);

/// Loads a model from a file.
/// - Parameter filename: The path to the file containing the model structure and config.
/// - Returns: A pointer to the loaded `Model` struct, or `NULL` if the file could not be loaded.
///
/// - Example Usage:
///     ```c
///     Model* model = load_model("model.json");
///     ```
///
Model* load_model(const char* filename);

/// Saves the current model structure to a `JSON` file.
/// - Parameters:
///   - model:    A pointer to the `Model` struct to be saved.
///   - filename: The path to the file where the model will be saved.
/// - Returns: 0 if the model was saved successfully, a non-zero value otherwise.
///
/// - Example Usage:
///     ```c
///     save_model(model, "model.json");
///     ```
///
int save_model(Model* model, const char* filename);

/// This function releases all dynamically allocated memory associated with the model, including layers, optimizer, loss and the model itself.
/// - Parameter model: Pointer to the `Model` to be freed.
///
/// - Example Usage:
///     ```c
///     free_model(model);
///     ```
///
void free_model(Model* model);

#endif /* cdnn_h */
