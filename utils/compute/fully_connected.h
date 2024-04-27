/* utils/compute/fully_connected.h
 *
 * This file provides implementations for fully connected layers, which are an essential
 * component of neural networks. Fully connected layers are typically used towards the end
 * of a neural network architecture, following convolutional and pooling layers, to perform
 * high-level reasoning and classification tasks.
 *
 * Key functionalities include:
 *
 * 1. Forward propagation of input data through a fully connected layer, computing the
 *    output activations.
 * 2. Backward propagation of gradients through a fully connected layer, computing
 *    gradients for the input, weights, and biases.
 * 3. Updating the weights and biases of a fully connected layer using the specified
 *    optimization algorithm (e.g., SGD, Adam, RMSprop).
 *
 * This header file serves as a central hub for working with fully connected layers,
 * providing a consistent and intuitive interface for developers to incorporate these
 * layers into their neural network models.
 */


#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "../dimension.h"
#include "../../optimizer/optimizer.h"

/*
 * Performs the forward pass through a fully connected layer.
 *
 * Parameters:
 * - input: A 3D array containing the input data.
 * - input_shape: The dimensions of the input data.
 * - num_neurons: The number of neurons (output units) in the fully connected layer.
 * - weights: A 2D array containing the weights of the fully connected layer.
 * - biases: A 1D array containing the biases of the fully connected layer.
 *
 * Returns:
 * - A 1D array containing the output activations of the fully connected layer.
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * Dimensions input_dim = {4, 4, 64};
 * int num_neurons = 128;
 * float** fc_weights = ...; // Load or initialize fully connected weights
 * float* fc_biases = ...; // Load or initialize fully connected biases
 * float* output_data = fc_forward(input_data, input_dim, num_neurons, fc_weights, fc_biases);
 */
float* fc_forward(float*** input, Dimensions input_shape, int num_neurons, float** weights, float* biases);

/*
 * Performs the backward pass through a fully connected layer.
 *
 * Parameters:
 * - input: A 3D array containing the input data used during the forward pass.
 * - output_grad: A 1D array containing the gradients of the output with respect to the loss.
 * - input_grad: A 3D array to store the gradients of the input with respect to the loss.
 * - input_shape: The dimensions of the input data.
 * - num_neurons: The number of neurons (output units) in the fully connected layer.
 * - weights: A 2D array containing the weights of the fully connected layer.
 * - weight_grads: A 2D array to store the gradients of the weights with respect to the loss.
 * - bias_grads: A 1D array to store the gradients of the biases with respect to the loss.
 * - prev_layer_is_flatten: A flag indicating whether the previous layer is a flatten layer.
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * float* output_grad = ...; // Compute or load output gradients
 * Dimensions input_dim = {4, 4, 64};
 * int num_neurons = 128;
 * float** fc_weights = ...; // Load or initialize fully connected weights
 * float** weight_grads = allocate_2d_array(num_neurons, input_dim.height * input_dim.width * input_dim.channels);
 * float* bias_grads = allocate_1d_array(num_neurons);
 * float*** input_grad = allocate_3d_array(input_dim);
 * fc_backward(input_data, output_grad, input_grad, input_dim, num_neurons, fc_weights, weight_grads, bias_grads, 0);
 */
void fc_backward(float*** input, float** output_grad, float*** input_grad, Dimensions input_shape, int num_neurons, float** weights, float** weight_grads, float* bias_grads, int prev_layer_is_flatten);

/*
 * Updates the weights and biases of a fully connected layer using the specified optimization algorithm.
 *
 * Parameters:
 * - num_neurons: The number of neurons (output units) in the fully connected layer.
 * - fc_weights: A 2D array containing the weights of the fully connected layer.
 * - fc_grads: A 2D array containing the gradients of the weights with respect to the loss.
 * - biases: A 1D array containing the biases of the fully connected layer.
 * - bias_grads: A 1D array containing the gradients of the biases with respect to the loss.
 * - optimizer: A pointer to the Optimizer struct representing the optimization algorithm.
 * - layer_index: The index of the fully connected layer in the neural network model.
 *
 * Usage example:
 *
 * float** fc_weights = ...; // Load or initialize fully connected weights
 * float** fc_grads = ...; // Compute gradients for the fully connected weights
 * float* fc_biases = ...; // Load or initialize fully connected biases
 * float* fc_bias_grads = ...; // Compute gradients for the fully connected biases
 * Optimizer* opt = create_optimizer("Adam", 0.001, ...);
 * update_fc_weights(128, fc_weights, fc_grads, fc_biases, fc_bias_grads, opt, 0);
 */
void update_fc_weights(int num_neurons, float** fc_weights, float** fc_grads, float* biases, float* bias_grads, Optimizer* optimizer, int layer_index);

#endif /* FULLY_CONNECTED_H */
