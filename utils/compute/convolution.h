/* utils/compute/convolutional.h
 *
 * This file provides implementations for performing convolutional operations, which
 * are essential for Convolutional Neural Networks (CNNs). Convolution is a mathematical
 * operation that applies a set of learnable filters to the input data, enabling the
 * network to extract and learn hierarchical features from the input.
 *
 * Key functionalities include:
 *
 * 1. Forward propagation of input data through the convolutional layer, computing
 *    the output feature maps.
 * 2. Backward propagation of gradients through the convolutional layer, computing
 *    gradients for the input, weights, and biases.
 * 3. Updating the weights and biases of the convolutional layer using the specified
 *    optimization algorithm (e.g., SGD, Adam, RMSprop).
 *
 * This header file serves as a central hub for working with convolutional layers,
 * providing a consistent and intuitive interface for developers to incorporate these
 * fundamental operations into their CNN models.
 */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "../dimension.h"
#include "../../optimizer/optimizer.h"

/*
 * Performs the forward pass through a convolutional layer.
 *
 * Parameters:
 * - input: A 3D array containing the input data.
 * - input_shape: The dimensions of the input data.
 * - output_shape: The dimensions of the output data.
 * - num_filters: The number of filters in the convolutional layer.
 * - filter_size: The size of each filter (assumed to be square).
 * - stride: The stride value for the convolution operation.
 * - padding: The padding value for the convolution operation.
 * - weights: A 4D array containing the weights of the convolutional layer.
 * - biases: A 1D array containing the biases of the convolutional layer.
 *
 * Returns:
 * - A 3D array containing the output feature maps after the convolution operation.
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * Dimensions input_dim = {32, 32, 3};
 * Dimensions output_dim = {30, 30, 32};
 * float**** conv_weights = ...; // Load or initialize convolutional weights
 * float* conv_biases = ...; // Load or initialize convolutional biases
 * float*** output_data = conv_forward(input_data, input_dim, output_dim, 32, 3, 1, 1, conv_weights, conv_biases);
 */
float*** conv_forward(float*** input, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float* biases);

/*
 * Performs the backward pass through a convolutional layer.
 *
 * Parameters:
 * - input: A 3D array containing the input data used during the forward pass.
 * - output_grad: A 3D array containing the gradients of the output with respect to the loss.
 * - input_grad: A 3D array to store the gradients of the input with respect to the loss.
 * - input_shape: The dimensions of the input data.
 * - output_shape: The dimensions of the output data.
 * - num_filters: The number of filters in the convolutional layer.
 * - filter_size: The size of each filter (assumed to be square).
 * - stride: The stride value for the convolution operation.
 * - padding: The padding value for the convolution operation.
 * - weights: A 4D array containing the weights of the convolutional layer.
 * - weight_grads: A 4D array to store the gradients of the weights with respect to the loss.
 * - bias_grads: A 1D array to store the gradients of the biases with respect to the loss.
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * float*** output_grad = ...; // Compute or load output gradients
 * Dimensions input_dim = {32, 32, 3};
 * Dimensions output_dim = {30, 30, 32};
 * float**** conv_weights = ...; // Load or initialize convolutional weights
 * float**** weight_grads = allocate_4d_array(32, 3, 3, 3);
 * float* bias_grads = allocate_1d_array(32);
 * float*** input_grad = allocate_3d_array(input_dim);
 * conv_backward(input_data, output_grad, input_grad, input_dim, output_dim, 32, 3, 1, 1, conv_weights, weight_grads, bias_grads);
 */
void conv_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float**** weight_grads, float* bias_grads);

/*
 * Updates the weights and biases of a convolutional layer using the specified optimization algorithm.
 *
 * Parameters:
 * - num_filters: The number of filters in the convolutional layer.
 * - filter_size: The size of each filter (assumed to be square).
 * - input_channels: The number of input channels to the convolutional layer.
 * - conv_weights: A 4D array containing the weights of the convolutional layer.
 * - conv_grads: A 4D array containing the gradients of the weights with respect to the loss.
 * - biases: A 1D array containing the biases of the convolutional layer.
 * - bias_grads: A 1D array containing the gradients of the biases with respect to the loss.
 * - optimizer: A pointer to the Optimizer struct representing the optimization algorithm.
 * - layer_index: The index of the convolutional layer in the neural network model.
 *
 * Usage example:
 *
 * float**** conv_weights = ...; // Load or initialize convolutional weights
 * float**** conv_grads = ...; // Compute gradients for the convolutional weights
 * float* conv_biases = ...; // Load or initialize convolutional biases
 * float* conv_bias_grads = ...; // Compute gradients for the convolutional biases
 * Optimizer* opt = create_optimizer("Adam", 0.001, ...);
 * update_conv_weights(32, 3, 3, conv_weights, conv_grads, conv_biases, conv_bias_grads, opt, 0);
 */
void update_conv_weights(int num_filters, int filter_size, int input_channels, float**** conv_weights, float**** conv_grads, float* biases, float* bias_grads, Optimizer* optimizer, int layer_index);

#endif /* CONVOLUTION_H */
