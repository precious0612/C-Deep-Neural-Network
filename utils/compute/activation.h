/* utils/compute/activation.h */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../dimension.h"

void relu_forward(float*** input, float*** output, Dimensions input_shape);
void relu_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
void sigmoid_forward(float*** input, float*** output, Dimensions input_shape);
void sigmoid_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
void tanh_forward(float*** input, float*** output, Dimensions input_shape);
void tanh_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
void max_forward(float*** input, float*** output, Dimensions input_shape);
void max_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
void softmax_forward(float*** input, float*** output, Dimensions input_shape);
void softmax_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
void apply_activation(const char* activation, float*** input, float*** output, Dimensions input_shape, float*** output_grad, float*** input_grad);
void apply_activation_backward(const char* activation, float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);

#endif // /* ACTIVATION_H */ 