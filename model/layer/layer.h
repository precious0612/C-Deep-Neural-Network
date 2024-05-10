/* model/layer/layer.h 
 * 
 * This file defines the interface for the layer object in the CNN model.
 *
 * The layer object contains the necessary information to perform the forward
 * pass, backward pass, and update the parameters for a specific layer in the
 * model.
 *
 * This header file provides the following functionality:
 *
 * 1. Definition of layer types (e.g., convolutional, pooling, fully connected)
 * 2. Structure definitions for layer parameters, weights, and biases
 * 3. Function declarations for creating, initializing, and deleting layers
 * 4. Function declarations for forward and backward passes through a layer
 * 5. Function declarations for updating layer weights and resetting gradients
 * 6. Function declaration for computing the output shape of a layer
 */

#ifndef LAYER_H
#define LAYER_H

#include "../../utils/utils.h"
#include "../../optimizer/optimizer.h"

// Define enumeration for layer types
typedef enum {
    CONVOLUTIONAL,
    POOLING,
    FULLY_CONNECTED,
    DROPOUT,
    ACTIVATION,
    FLATTEN
} LayerType;

// Define structure for holding layer parameters
typedef union {
    struct {
        int num_filters;
        int filter_size;
        int stride;
        int padding;
        char activation[20];
    } conv_params;
    struct {
        int pool_size;
        int stride;
        char pool_type[10];
    } pooling_params;
    struct {
        int num_neurons;
        char activation[20];
    } fc_params;
    struct {
        float dropout_rate;
    } dropout_params;
    struct {
        char activation[20];
    } activation_params;
} LayerParams;

typedef union {
    float**** conv_weights;     // Weights for convolutional layers
    float**** conv_grads;       // Gradients for convolutional weights
    float*** pool_weights;      // Weights for pooling layers (not typically used)
    float** fc_weights;         // Weights for fully connected layers
    float** fc_grads;           // Gradients for fully connected weights
    float* dropout_weights;     // Weights for dropout layers (not typically used)
} Weights;

typedef struct {
    float* biases;      // Biases for the layer
    float* bias_grads;  // Gradients for biases
} LayerBiases;

// Define structure for representing a layer in the CNN model
typedef struct Layer {
    LayerType type;           // Type of the layer
    LayerParams params;       // Parameters specific to the layer type
    Weights weights;          // Weights for the layer
    Weights grads;            // Gradients for the layer weights
    LayerBiases biases;       // Biases and bias gradients for the layer
    Dimensions input_shape;   // Shape of the input data
    Dimensions output_shape;  // Shape of the output data
    struct Layer* next_layer; // Pointer to the next layer in the model
    struct Layer* prev_layer; // Pointer to the previous layer in the model
} Layer;

/*
 * Creates a new layer of the specified type with the given parameters.
 *
 * Parameters:
 * - type: The type of the layer (e.g., CONVOLUTIONAL, POOLING, FULLY_CONNECTED).
 * - params: The parameters specific to the layer type.
 * 
 * Returns:
 * - A pointer to the newly created Layer struct, or NULL if memory allocation fails.
 *
 * Tips: The returned layer must be initialized using the initialize_layer function before use.
 *
 * Usage example:
 *
 * LayerParams conv_params = {
 *     .conv_params = {
 *         .num_filters = 32,
 *         .filter_size = 3,
 *         .stride = 1,
 *         .padding = 1,
 *         .activation = "relu"
 *     }
 * };
 * Layer* conv_layer = create_layer(CONVOLUTIONAL, conv_params);
 * if (conv_layer == NULL) {
 *     // Handle error
 * }
 * initialize_layer(conv_layer);
 */
Layer* create_layer(LayerType type, LayerParams params);

/*
 * Initializes the weights and biases for the given layer.
 *
 * Parameters:
 * - layer: A pointer to the Layer struct to be initialized.
 *
 * Usage example:
 *
 * Layer* conv_layer = create_layer(CONVOLUTIONAL, conv_params);
 * initialize_layer(conv_layer);
 */
void initialize_layer(Layer* layer);

/*
 * Performs the forward pass through the given layer.
 *
 * Parameters:
 * layer: A pointer to the Layer struct.
 * input: A 3D array containing the input data for the layer.
 * 
 * Returns:
 * - A 3D array containing the output data from the layer.
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * float*** output_data = layer_forward_pass(conv_layer, input_data);
 * if (output_data == NULL) {
 *     // Handle error
 * }
 * // Process output_data
 * free_3d_array(output_data, conv_layer->output_shape);
 */
float*** layer_forward_pass(Layer* layer, float*** input);

/*
 * Performs the backward pass through the given layer.
 *
 * Parameters:
 * - layer: A pointer to the Layer struct.
 * - input: A 3D array containing the input data for the layer.
 * - output_grad: A 3D array containing the gradient of the output from the layer.
 * - input_grad: A 3D array to store the gradient of the input to the layer.
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * float*** output_grad = ...; // Compute or load output gradient
 * float*** input_grad = allocate_3d_array(layer->input_shape);
 * layer_backward_pass(conv_layer, input_data, output_grad, input_grad);
 * // Process input_grad
 * free_3d_array(input_grad, conv_layer->input_shape);
 */
void layer_backward_pass(Layer* layer, float*** input, float*** output_grad, float*** input_grad);

/*
 * Updates the weights and biases of the given layer using the specified optimizer.
 *
 * Parameters:
 * - layer: A pointer to the Layer struct.
 * - optimizer: A pointer to the Optimizer struct to be used for updating the weights and biases.
 * - layer_index: The index of the layer in the neural network model.
 * Tips: Optimizer is a struct that contains the optimizer type and hyperparameters, 
 *       which is defined in optimizer/optimizer.h.
 *
 * Usage example:
 *
 * num_weights = conv_layer->output_shape.depth * conv_layer->output_shape.height * conv_layer->output_shape.width;
 * Optimizer* optimizer = create_optimizer("Adam", 0.001, num_weights);
 * update_layer_weights(conv_layer, optimizer, 0); // Update weights for the first layer
 * delete_optimizer(optimizer);
 */
void update_layer_weights(Layer* layer, Optimizer* optimizer, int layer_index);

/*
 * Resets the gradients for the given layer to zero.
 *
 * Parameters:
 * - layer: A pointer to the Layer struct.
 *
 * Usage example:
 *
 * reset_layer_grads(conv_layer);
 */
void reset_layer_grads(Layer* layer);

/*
 * Computes the output shape of the given layer based on its input shape and parameters.
 *
 * Parameters:
 * - layer A pointer to the Layer struct.
 *
 * Usage example:
 *
 * compute_output_shape(conv_layer);
 * Dimensions output_shape = conv_layer->output_shape;
 */
void compute_output_shape(Layer* layer);

/*
 * Deletes the given layer, freeing the memory allocated for its weights and biases.
 *
 * Parameters:
 * - layer: A pointer to the Layer struct to be deleted.
 *
 * Usage example:
 *
 * delete_layer(conv_layer);
 */
void delete_layer(Layer* layer);

void save_conv_weights(hid_t group_id, float ****weights, int num_filters, int filter_size, int channels);

#endif /* LAYER_H */
