//
//  layer.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/18/24.
//

#ifndef layer_h
#define layer_h

#include "../../utils/utils.h"
#include "../optimizer/optimizer.h"

// MARK: - Define the structures of layer

// TODO: Define enumeration for layer types
typedef enum {
    CONVOLUTIONAL,
    POOLING,
    FULLY_CONNECTED,
    DROPOUT,
    ACTIVATION,
    FLATTEN,
    LAYER_TYPE_COUNT
} LayerType;

// TODO: Define structure for holding layer parameters
typedef union {
    struct {
        int num_filters;
        int filter_size;
        int stride;
        int padding;
        ActivationType activation;
    } conv_params;
    struct {
        int pool_size;
        int stride;
        PoolType pool_type;
    } pooling_params;
    struct {
        int num_neurons;
        ActivationType activation;
    } fc_params;
    struct {
        float dropout_rate;
    } dropout_params;
    struct {
        ActivationType activation;
    } activation_params;
} LayerParams;

// TODO: Define union for different layer type weights and biases
typedef union {
    float**** conv_weights;    // shape: {output_channels, input_channels, width, height}
    float**   fc_weights;      // shape: {num_neurons, input_size}
} LayerWeights;

typedef float* LayerBiases;

// TODO: Define union for different layer type grads
typedef union {
    float**** conv_grads;    // shape: {output_channels, input_channels, width, height}
    float**   fc_grads;      // shape: {num_neurons, input_size}
} LayerGrads;

typedef float* LayerBiasGrads;

// TODO: Define structure for representing a layer in the model
typedef float*** LayerInput;
typedef float*** LayerOutput;
typedef struct Layer {
    LayerType type;            // Type of the layer
    LayerParams params;        // Parameters specific to the layer type
    LayerWeights weights;      // Weights for the layer
    LayerGrads grads;          // Gradients for the layer weights
    LayerBiases biases;        // Biases and bias gradients for the layer
    LayerBiasGrads bias_grads; // Gradients for the layer biases
    int num_params;
    Dimensions input_shape;    // Shape of the input data
    Dimensions output_shape;   // Shape of the output data
    LayerInput input;          // Input data from the layer
    LayerInput input_before_activation;
    struct Layer* next_layer;  // Pointer to the next layer in the model
    struct Layer* prev_layer;  // Pointer to the previous layer in the model
} Layer;

typedef float*** LayerInputGrad;
typedef float*** LayerOutputGrad;

// MARK: - Method Declarations

/// Creates a new layer of the specified type with the given parameters.
/// - Parameters:
///   - type: The type of the layer (e.g., `CONVOLUTIONAL`, `POOLING`, `FULLY_CONNECTED`).
///   - params: The parameters specific to the layer type.
/// - Returns: A pointer to the newly created `Layer` struct, or `NULL` if memory allocation fails.
///
/// - Tips: The returned layer must be initialized using the `initialize_layer` function before use.
///
/// - Example Usage:
///     ```c
///     LayerParams conv_params = {
///         .conv_params = {
///             .num_filters = 32,
///             .filter_size = 3,
///             .stride = 1,
///             .padding = 1,
///             .activation = "relu"
///         }
///     };
///     Layer* conv_layer = create_layer(CONVOLUTIONAL, conv_params);
///     ```
///
Layer* create_layer(LayerType type, LayerParams params);

/// Initializes the weights and biases for the given layer.
/// - Parameter layer: A pointer to the `Layer` struct to be initialized.
///
void initialize_layer(Layer* layer);

/// Performs the forward pass through the given layer.
/// - Parameters:
///   - layer: A pointer to the `Layer` struct.
///   - input: A 3D array containing the input data for the layer.
///   - is_training: If training, store the layer input. (1 is training, 0 is not.)
/// - Returns: A `LayerOutput` containing the output data from the layer.
///
/// - Example Usage:
///     ```c
///     float*** input_data = ...; // Load or generate input data
///     float*** output_data = layer_forward_pass(conv_layer, input_data, 0);
///     if (output_data == NULL) {
///         // Handle error
///     }
///     ```
///
LayerOutput layer_forward_pass(Layer* layer, LayerInput input, int is_training);

/// Performs the backward pass through the given layer.
/// - Parameters:
///   - layer: A pointer to the `Layer` struct.
///   - input: A `LayerInput` containing the input data for the layer.
///   - input_grad: A `LayerInputGrad` containing the gradient of the output from the layer.
///   - output_grad: A `LayerOutputGrad` to store the gradient of the input to the layer.
///
/// - Example Usage:
///     ```c
///     float*** input_data = ...; // Load or generate input data
///     float*** output_grad = ...; // Compute or load output gradient
///     float*** input_grad = calloc_float_3d_array(layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels);
///     layer_backward_pass(conv_layer, input_data, output_grad, input_grad);
///     ```
///
void layer_backward_pass(Layer* layer, LayerInputGrad input_grad, LayerOutputGrad output_grad);

/// Updates the weights and biases of the given layer using the specified optimizer.
/// - Parameters:
///   - layer: A pointer to the `Layer` struct.
///   - optimizer: A pointer to the `Optimizer` struct to be used for updating the weights and biases.
///   - layer_index: The index of the layer in the neural network model.
///
/// - Example Usage:
///     ```c
///     num_weights = conv_layer->output_shape.depth * conv_layer->output_shape.height * conv_layer->output_shape.width;
///     Optimizer* optimizer = create_optimizer("Adam", 0.001, num_weights);
///     update_layer_weights(conv_layer, optimizer, 0); // Update weights for the first layer
///     ```
///
void update_layer_weights(Layer* layer, Optimizer* optimizer, int layer_index);

/// Resets the gradients for the given layer to zero.
/// - Parameter layer: A pointer to the `Layer` struct.
///
/// - Example Usage:
///     ```c
///     reset_layer_grads(conv_layer);
///     ```
///
void reset_layer_grads(Layer* layer);

/// Resets the gradients for the given layer to zero.
/// - Parameter layer: A pointer to the `Layer` struct.
///
/// - Example Usage:
///     ```c
///     compute_output_shape(conv_layer);
///     ```
///
void compute_output_shape(Layer* layer);

/// Deletes the given layer, freeing the memory allocated for its weights and biases.
/// - Parameter layer: A pointer to the `Layer` struct to be deleted.
///
/// - Example Usage:
///     ```c
///     delete_layer(conv_layer);
///     ```
///     
void delete_layer(Layer* layer);

#endif /* layer_h */
