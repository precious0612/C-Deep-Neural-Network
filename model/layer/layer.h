/* model/layer/layer.h 
 *
 * This file defines the interface for the layer object in the CNN model.
 * The layer object contains the necessary information to perform the forward
 * pass, backward pass, and update the parameters for a specific layer in the
 * model.
 *
 */

#ifndef LAYER_H
#define LAYER_H

#include "../../utils/utils.h"

// Define enumeration for layer types
typedef enum {
    CONVOLUTIONAL,
    POOLING,
    FULLY_CONNECTED,
    DROPOUT
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
} LayerParams;

typedef union {
    float**** conv_weights;     // Weights for convolutional layers
    float*** pool_weights;      // Weights for pooling layers (not typically used)
    float** fc_weights;         // Weights for fully connected layers
    float* dropout_weights;     // Weights for dropout layers (not typically used)
} Weights;

// Define structure for representing a layer in the CNN model
typedef struct Layer {
    LayerType type;           // Type of the layer
    LayerParams params;       // Parameters specific to the layer type
    Weights weights;          // Weights for the layer
    float* biases;            // Biases for the layer
    Dimensions input_shape;   // Shape of the input data
    Dimensions output_shape;  // Shape of the output data
    struct Layer* next_layer; // Pointer to the next layer in the model
    struct Layer* prev_layer; // Pointer to the previous layer in the model
} Layer;

// Define function to create a new layer
Layer* create_layer(LayerType type, LayerParams params);

// Define function to initialize the weights and biases for a layer
void initialize_layer(Layer* layer);

// Define function to forward pass through a layer
void layer_forward_pass(Layer* layer, float*** input, float*** output);

// Define function to backward pass through a layer
void layer_backward_pass(Layer* layer, float*** input, float*** output_grad, float*** input_grad, float learning_rate);

// Define function to compute the output shape
void compute_output_shape(Layer* layer);

// Define function to delete a layer
void delete_layer(Layer* layer);

#endif /* LAYER_H */
