/* utils/compute/flatten.h
 *
 * This file provides implementations for the flatten and unflatten operations, which
 * are commonly used in neural networks. The flatten operation transforms a multi-dimensional
 * tensor into a one-dimensional vector, while the unflatten operation performs the reverse
 * transformation, converting a one-dimensional vector back into a multi-dimensional tensor.
 *
 * These operations are essential for transitioning between convolutional layers and fully
 * connected layers in a Convolutional Neural Network (CNN). By flattening the output of
 * convolutional layers, the network can process the hierarchical features learned by the
 * convolutional layers through fully connected layers.
 *
 * Key functionalities include:
 *
 * 1. Flattening a multi-dimensional tensor into a one-dimensional vector.
 * 2. Unflattening a one-dimensional vector into a multi-dimensional tensor.
 *
 * This header file serves as a central hub for working with flatten and unflatten operations,
 * providing a consistent and intuitive interface for developers to incorporate these transformations
 * into their neural network models.
 */


#ifndef FLATTEN_H
#define FLATTEN_H

#include "../dimension.h"

/*
 * Flattens a multi-dimensional tensor into a one-dimensional vector.
 *
 * Parameters:
 * - input: A 3D array containing the input tensor.
 * - flattened: A 1D array to store the flattened vector.
 * - input_shape: The dimensions of the input tensor.
 *
 * Usage example:
 *
 * float*** input_tensor = ...; // Load or generate input tensor
 * Dimensions input_dim = {4, 4, 3};
 * float* flattened_vector = allocate_1d_array(input_dim.height * input_dim.width * input_dim.channels);
 * flatten(input_tensor, flattened_vector, input_dim);
 */
void flatten(float*** input, float* flattened, Dimensions input_shape);

/*
 * Unflattens a one-dimensional vector into a multi-dimensional tensor.
 *
 * Parameters:
 * - flattened: A 1D array containing the flattened vector.
 * - output: A 3D array to store the unflattened tensor.
 * - output_shape: The dimensions of the output tensor.
 *
 * Usage example:
 *
 * float* flattened_vector = ...; // Load or generate flattened vector
 * Dimensions output_dim = {4, 4, 3};
 * float*** output_tensor = allocate_3d_array(output_dim);
 * unflatten(flattened_vector, output_tensor, output_dim);
 */
void unflatten(float* flattened, float*** output, Dimensions output_shape);

#endif // /* FLATTEN_ */