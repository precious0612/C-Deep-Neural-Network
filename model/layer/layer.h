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

// Define structure for representing a layer in the CNN model
typedef struct Layer {
    LayerType type;           // Type of the layer
    LayerParams params;       // Parameters specific to the layer type
    float** weights;          // Weights for the layer
    float* biases;            // Biases for the layer
    struct Layer* next_layer; // Pointer to the next layer in the model
} Layer;

#endif /* LAYER_H */
