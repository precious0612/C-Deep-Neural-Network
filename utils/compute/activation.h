/* utils/compute/activation.h */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../dimension.h"

float*** relu_forward(float*** input, Dimensions input_shape);
void relu_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
float*** sigmoid_forward(float*** input, Dimensions input_shape);
void sigmoid_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
float*** tanh_forward(float*** input, Dimensions input_shape);
void tanh_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
float*** max_forward(float*** input, Dimensions input_shape);
void max_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
float*** softmax_forward(float*** input, Dimensions input_shape);
void softmax_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);
float*** apply_activation(const char* activation, float*** input, Dimensions input_shape);
void apply_activation_backward(const char* activation, float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);

#endif // /* ACTIVATION_H */ 