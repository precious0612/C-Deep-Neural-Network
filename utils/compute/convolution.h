/* utils/compute/convolutional.h */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

void conv_forward(float*** input, float*** output, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float* biases);
void conv_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float* biases, float learning_rate);

#endif /* CONVOLUTION_H */
