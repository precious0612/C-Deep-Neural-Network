//
//  convolution.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/19/24.
//

#ifndef convolution_h
#define convolution_h

float*** conv_forward(float*** input, float**** weights, float* biases, int input_width, int input_height, int input_channels, int num_filters, int filter_size, int stride, int padding);

void conv_backward(float*** input, float**** weights, float*** output_grad, float*** input_grad, float**** weight_grads, float* bias_grads, int input_width, int input_height, int input_channels, int num_filters, int filter_size, int stride, int padding);

#endif /* convolution_h */
