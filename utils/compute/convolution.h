/* utils/compute/convolutional.h */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "../dimension.h"
#include "../../optimizer/optimizer.h"

float*** conv_forward(float*** input, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float* biases);
void conv_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float**** weight_grads, float* bias_grads);
void update_conv_weights(int num_filters, int filter_size, int input_channels, float**** conv_weights, float**** conv_grads, float* biases, float* bias_grads, Optimizer* optimizer, int layer_index);

#endif /* CONVOLUTION_H */
