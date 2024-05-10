/* utils/compute/activation.h
 *
 * This file provides implementations for various activation functions commonly used
 * in neural networks, such as ReLU, Sigmoid, Tanh, Max, and Softmax. These activation
 * functions introduce non-linearities into the network, enabling it to learn complex
 * patterns and relationships from the input data.
 *
 * Key functionalities include:
 *
 * 1. Forward propagation of input data through different activation functions.
 * 2. Backward propagation of gradients through the respective activation functions.
 * 3. A unified interface for applying activation functions during the forward pass.
 * 4. A unified interface for computing gradients during the backward pass.
 *
 * This header file serves as a central hub for working with various activation functions,
 * providing a consistent and intuitive interface for developers to incorporate these
 * non-linearities into their neural network models.
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../dimension.h"

/*
 * Performs the forward pass of the activation functions.
 *
 * Parameters:
 * - input: A 3D array containing the input data.
 * - input_shape: The dimensions of the input data.
 *
 * Returns:
 * - A 3D array containing the output data after applying the ReLU activation.
 *
 * Usage example:
 *
 * float*** input_data = ...; // Load or generate input data
 * Dimensions input_dim = {32, 32, 3};
 * float*** output_data = ***_forward(input_data, input_dim);
 */
float*** relu_forward(float*** input, Dimensions input_shape);
float*** sigmoid_forward(float*** input, Dimensions input_shape);
float*** tanh_forward(float*** input, Dimensions input_shape);
float*** max_forward(float*** input, Dimensions input_shape);
float*** softmax_forward(float*** input, Dimensions input_shape);

/*
 * Performs the backward pass of the activation functions.
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
 * Dimensions input_dim = {32, 32, 3};
 * float*** input_grad = allocate_3d_array(input_dim);
 * ***_backward(input_data, output_grad, input_grad, input_dim);
 */
void relu_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
void sigmoid_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
void tanh_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
void max_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
void softmax_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);

/*
 * Applies the specified activation function to the input data.
 *
 * Parameters:
 * - activation: A string specifying the activation function to apply (e.g., "relu", "sigmoid").
 * - input: A 3D array containing the input data.
 * - input_shape: The dimensions of the input data.
 *
 * Returns:
 * - A 3D array containing the output data after applying the specified activation function.
 *
 * Usage example:
 *
 * const char* activation = "relu";
 * float*** input_data = ...; // Load or generate input data
 * Dimensions input_dim = {32, 32, 3};
 * float*** output_data = forward_activation(activation, input_data, input_dim);
 */
float*** forward_activation(const char* activation, float*** input, Dimensions input_shape);

/*
 * Computes the gradients of the input data with respect to the loss for the specified activation function.
 *
 * Parameters:
 * - activation: A string specifying the activation function to apply (e.g., "relu", "sigmoid").
 * - input: A 3D array containing the input data used during the forward pass.
 * - output_grad: A 3D array containing the gradients of the output with respect to the loss.
 * - input_grad: A 3D array to store the gradients of the input with respect to the loss.
 * - input_shape: The dimensions of the input data.
 *
 * Usage example:
 *
 * const char* activation = "relu";
 * float*** input_data = ...; // Load or generate input data
 * float*** output_grad = ...; // Compute or load output gradients
 * Dimensions input_dim = {32, 32, 3};
 * float*** input_grad = allocate_3d_array(input_dim);
 * backward_activation(activation, input_data, output_grad, input_grad, input_dim);
 */
void backward_activation(const char* activation, float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);

#endif // /* ACTIVATION_H */ 