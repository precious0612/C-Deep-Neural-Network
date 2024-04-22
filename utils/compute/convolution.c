/* utils/compute/convolution.c */

#include <stdlib.h>
#include <string.h>

#include "convolution.h"
#include "../memory.h"
#include "../dimension.h"

void conv_forward(float*** input, float*** output, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float* biases) {

    // Iterate over output elements
    for (int out_ch = 0; out_ch < num_filters; out_ch++) {
        for (int out_y = 0; out_y < output_shape.height; out_y++) {
            for (int out_x = 0; out_x < output_shape.width; out_x++) {
                float sum = 0.0f;
                // Perform convolution operation
                for (int in_ch = 0; in_ch < input_shape.channels; in_ch++) {
                    for (int fy = 0; fy < filter_size; fy++) {
                        for (int fx = 0; fx < filter_size; fx++) {
                            int in_y = out_y * stride + fy - padding;
                            int in_x = out_x * stride + fx - padding;
                            if (in_y >= 0 && in_y < input_shape.height && in_x >= 0 && in_x < input_shape.width) {
                                sum += input[in_y][in_x][in_ch] * weights[out_ch][fy][fx][in_ch];
                            }
                        }
                    }
                }
                output[out_y][out_x][out_ch] = sum + biases[out_ch];
            }
        }
    }
}

void conv_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float* biases, float learning_rate) {

    // Compute gradients for the input
    for (int out_ch = 0; out_ch < num_filters; out_ch++) {
        for (int out_y = 0; out_y < output_shape.height; out_y++) {
            for (int out_x = 0; out_x < output_shape.width; out_x++) {
                for (int in_ch = 0; in_ch < input_shape.channels; in_ch++) {
                    for (int fy = 0; fy < filter_size; fy++) {
                        for (int fx = 0; fx < filter_size; fx++) {
                            int in_y = out_y * stride + fy - padding;
                            int in_x = out_x * stride + fx - padding;
                            if (in_y >= 0 && in_y < input_shape.height && in_x >= 0 && in_x < input_shape.width) {
                                input_grad[in_y][in_x][in_ch] += output_grad[out_y][out_x][out_ch] * weights[out_ch][fy][fx][in_ch];
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute gradients for the weights and biases
     float**** weight_grad = malloc_4d_float_array(num_filters, filter_size, filter_size, input_shape.channels);
    float* bias_grad = calloc(num_filters, sizeof(float));

    for (int out_ch = 0; out_ch < num_filters; out_ch++) {
        for (int out_y = 0; out_y < output_shape.height; out_y++) {
            for (int out_x = 0; out_x < output_shape.width; out_x++) {
                bias_grad[out_ch] += output_grad[out_y][out_x][out_ch];
                for (int in_ch = 0; in_ch < input_shape.channels; in_ch++) {
                    for (int fy = 0; fy < filter_size; fy++) {
                        for (int fx = 0; fx < filter_size; fx++) {
                            int in_y = out_y * stride + fy - padding;
                            int in_x = out_x * stride + fx - padding;
                            if (in_y >= 0 && in_y < input_shape.height && in_x >= 0 && in_x < input_shape.width) {
                                weight_grad[out_ch][fy][fx][in_ch] += input[in_y][in_x][in_ch] * output_grad[out_y][out_x][out_ch];
                            }
                        }
                    }
                }
            }
        }
    }

     // Update weights and biases
    for (int out_ch = 0; out_ch < num_filters; out_ch++) {
        biases[out_ch] -= learning_rate * bias_grad[out_ch];
        for (int fy = 0; fy < filter_size; fy++) {
            for (int fx = 0; fx < filter_size; fx++) {
                for (int in_ch = 0; in_ch < input_shape.channels; in_ch++) {
                    float new_weight = weights[out_ch][fy][fx][in_ch] - learning_rate * weight_grad[out_ch][fy][fx][in_ch];
                    weights[out_ch][fy][fx][in_ch] = new_weight;
                }
            }
        }
    }

    // Free memory for gradients
    free_4d_float_array(weight_grad, num_filters, filter_size, filter_size, input_shape.channels);
    free(bias_grad);
}