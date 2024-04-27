/* utils/compute/dropout.h
 *
 * This file provides implementations for the dropout regularization technique, which
 * is used to prevent overfitting in neural networks. Dropout randomly sets a fraction
 * of input units to zero during the forward pass, effectively creating a different
 * network architecture for each pass and allowing the network to learn more robust
 * features.
 *
 * Key functionalities include:
 *
 * 1. Forward propagation of input data through the dropout layer, randomly setting
 *    a fraction of input units to zero.
 * 2. Backward propagation of gradients through the dropout layer, propagating the
 *    gradients only for the input units that were not set to zero during the forward pass.
 *
 * This header file serves as a central hub for working with the dropout regularization
 * technique, providing a consistent and intuitive interface for developers to incorporate
 * dropout into their neural network models.
 */

#ifndef DROPOUT_H
#define DROPOUT_H

#include "../dimension.h"

/*
 * Performs the forward pass through a dropout layer.
 *
 * Parameters:
 * - input: A 3D array containing the input data.
 * - input_shape: The dimensions of the input data.
 * - dropout_rate: The fraction of input units to set to zero.
 *
 * Returns:
 * - A 3D array containing the output data after applying dropout.
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * Dimensions input_dim = {32, 32, 64};
 * float dropout_rate = 0.5;
 * float*** output_data = dropout_forward(input_data, input_dim, dropout_rate);
 */
float*** dropout_forward(float*** input, Dimensions input_shape, float dropout_rate);

/*
 * Performs the backward pass through a dropout layer.
 *
 * Parameters:
 * - input: A 3D array containing the input data used during the forward pass.
 * - output_grad: A 3D array containing the gradients of the output with respect to the loss.
 * - input_grad: A 3D array to store the gradients of the input with respect to the loss.
 * - input_shape: The dimensions of the input data.
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * float*** output_grad = ...; // Compute or load output gradients
 * Dimensions input_dim = {32, 32, 64};
 * float*** input_grad = allocate_3d_array(input_dim);
 * dropout_backward(input_data, output_grad, input_grad, input_dim);
 */
void dropout_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);

#endif /* DROPOUT_H */