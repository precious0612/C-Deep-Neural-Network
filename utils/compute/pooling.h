/* utils/compute/pooling.h
 *
 * This file provides implementations for pooling operations, which are an essential
 * component of Convolutional Neural Networks (CNNs). Pooling is a technique used to
 * downsample the spatial dimensions of feature maps, reducing computational complexity
 * and introducing translation invariance.
 *
 * Key functionalities include:
 *
 * 1. Forward propagation of input data through a pooling layer, applying either
 *    max pooling or average pooling operation.
 * 2. Backward propagation of gradients through a pooling layer, computing gradients
 *    for the input based on the pooling operation.
 *
 * This header file serves as a central hub for working with pooling operations,
 * providing a consistent and intuitive interface for developers to incorporate
 * these techniques into their CNN models.
 */

#ifndef POOLING_H
#define POOLING_H

#include "../dimension.h"

/*
 * Performs the forward pass through a pooling layer.
 *
 * Parameters:
 * - input: A 3D array containing the input data.
 * - input_shape: The dimensions of the input data.
 * - output_shape: The dimensions of the output data.
 * - pool_size: The size of the pooling window (assumed to be square).
 * - stride: The stride value for the pooling operation.
 * - pool_type: A string representing the type of pooling ("max" or "avg").
 *
 * Returns:
 * - A 3D array containing the output feature maps after the pooling operation.
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * Dimensions input_dim = {32, 32, 64};
 * Dimensions output_dim = {16, 16, 64};
 * float*** output_data = pool_forward(input_data, input_dim, output_dim, 2, 2, "max");
 */
float*** pool_forward(float*** input, Dimensions input_shape, Dimensions output_shape, int pool_size, int stride, char* pool_type);

/*
 * Performs the backward pass through a pooling layer.
 *
 * Parameters:
 * - input: A 3D array containing the input data used during the forward pass.
 * - output_grad: A 3D array containing the gradients of the output with respect to the loss.
 * - input_grad: A 3D array to store the gradients of the input with respect to the loss.
 * - input_shape: The dimensions of the input data.
 * - output_shape: The dimensions of the output data.
 * - pool_size: The size of the pooling window (assumed to be square).
 * - stride: The stride value for the pooling operation.
 * - pool_type: A string representing the type of pooling ("max" or "avg").
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * float*** output_grad = ...; // Compute or load output gradients
 * Dimensions input_dim = {32, 32, 64};
 * Dimensions output_dim = {16, 16, 64};
 * float*** input_grad = allocate_3d_array(input_dim);
 * pool_backward(input_data, output_grad, input_grad, input_dim, output_dim, 2, 2, "max");
 */
void pool_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape, Dimensions output_shape, int pool_size, int stride, char* pool_type);

#endif /* POOLING_H */
