/* utils/tensor.h
 *
 * This file provides utility functions for working with tensors, which are
 * multi-dimensional arrays commonly used in deep learning applications to represent
 * data and model parameters. These functions facilitate memory allocation, deallocation,
 * and manipulation of tensors, ensuring efficient and consistent handling of these
 * data structures throughout the codebase.
 *
 * Key functionalities include:
 *
 * 1. Allocating memory for output tensors and gradient tensors.
 * 2. Copying the contents of a tensor to a new tensor.
 * 3. Freeing the memory allocated for a tensor.
 *
 * By providing a centralized set of functions for tensor operations, this header file
 * simplifies the development and maintenance of deep learning models, enabling developers
 * to focus on the core logic while ensuring proper memory management and data handling.
 *
 * Usage examples:
 *
 * // Allocate memory for an output tensor
 * Dimensions output_shape = {10, 10, 32};
 * float*** output_tensor = allocate_output_tensor(output_shape);
 *
 * // Allocate memory for a gradient tensor
 * Dimensions input_shape = {28, 28, 1};
 * float*** input_grad = allocate_grad_tensor(input_shape);
 *
 * // Copy the contents of a tensor to a new tensor
 * float*** input_tensor = ...; // Load or generate input tensor
 * float*** copied_tensor = copy_3d_array(input_tensor, input_shape);
 *
 * // Free the memory allocated for a tensor
 * free_tensor(output_tensor, output_shape);
 */

#ifndef TENSOR_H
#define TENSOR_H

#include "dimension.h"

float*** allocate_output_tensor(Dimensions output_shape);
float*** allocate_grad_tensor(Dimensions shape);
float*** copy_3d_array(float*** src, Dimensions shape);
void free_tensor(float*** tensor, Dimensions tensor_shape);

#endif /* TENSOR_H */