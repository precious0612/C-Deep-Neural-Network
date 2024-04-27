/* utils/compute/convolution.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "convolution.h"
#include "../optim.h"
#include "../memory.h"

float*** conv_forward(float*** input, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float* biases) {
    // Allocate memory for the output
    float*** output = malloc_3d_float_array(output_shape.height, output_shape.width, num_filters);

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

    return output;
}

void conv_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float**** weight_grads, float* bias_grads) {
    // Initialize weight and bias gradients to zero
    for (int f = 0; f < num_filters; f++) {
        for (int h = 0; h < filter_size; h++) {
            for (int w = 0; w < filter_size; w++) {
                for (int c = 0; c < input_shape.channels; c++) {
                    weight_grads[f][h][w][c] = 0.0f;
                }
            }
        }
        bias_grads[f] = 0.0f;
    }

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
    for (int out_ch = 0; out_ch < num_filters; out_ch++) {
        for (int out_y = 0; out_y < output_shape.height; out_y++) {
            for (int out_x = 0; out_x < output_shape.width; out_x++) {
                bias_grads[out_ch] += output_grad[out_y][out_x][out_ch];
                for (int in_ch = 0; in_ch < input_shape.channels; in_ch++) {
                    for (int fy = 0; fy < filter_size; fy++) {
                        for (int fx = 0; fx < filter_size; fx++) {
                            int in_y = out_y * stride + fy - padding;
                            int in_x = out_x * stride + fx - padding;
                            if (in_y >= 0 && in_y < input_shape.height && in_x >= 0 && in_x < input_shape.width) {
                                weight_grads[out_ch][fy][fx][in_ch] += input[in_y][in_x][in_ch] * output_grad[out_y][out_x][out_ch];
                            }
                        }
                    }
                }
            }
        }
    }
}

void update_conv_weights(int num_filters, int filter_size, int input_channels, float**** conv_weights, float**** conv_grads, float* biases, float* bias_grads, Optimizer* optimizer, int layer_index) {
    int num_params = num_filters * filter_size * filter_size * input_channels;
    int counter = 0;

    switch (optimizer->type) {
        case SGD:
            for (int i = 0; i < num_filters; i++) {
                for (int j = 0; j < filter_size; j++) {
                    for (int k = 0; k < filter_size; k++) {
                        for (int l = 0; l < input_channels; l++) {
                            conv_weights[i][j][k][l] -= sgd(conv_grads[i][j][k][l], optimizer->optimizer.sgd->momentum, optimizer->optimizer.sgd->momentum_buffer[layer_index][i], optimizer->optimizer.sgd->learning_rate);
                        }
                    }
                }
                biases[i] -= sgd(bias_grads[i], optimizer->optimizer.sgd->momentum, optimizer->optimizer.sgd->momentum_buffer[layer_index][num_params + i], optimizer->optimizer.sgd->learning_rate);
            }
            break;
        case ADAM:
            for (int i = 0; i < num_filters; i++) {
                for (int j = 0; j < filter_size; j++) {
                    for (int k = 0; k < filter_size; k++) {
                        for (int l = 0; l < input_channels; l++) {
                            conv_weights[i][j][k][l] -= adam(conv_grads[i][j][k][l], optimizer->optimizer.adam->t, optimizer->optimizer.adam->m[layer_index][counter], optimizer->optimizer.adam->v[layer_index][counter], optimizer->optimizer.adam->beta1, optimizer->optimizer.adam->beta2, optimizer->optimizer.adam->epsilon, optimizer->optimizer.adam->learning_rate);
                            counter++;
                        }
                    }
                }
                biases[i] -= adam(bias_grads[i], optimizer->optimizer.adam->t, optimizer->optimizer.adam->m[layer_index][num_params + i], optimizer->optimizer.adam->v[layer_index][num_params + i], optimizer->optimizer.adam->beta1, optimizer->optimizer.adam->beta2, optimizer->optimizer.adam->epsilon, optimizer->optimizer.adam->learning_rate);
            }
            optimizer->optimizer.adam->t++;
            break;
        case RMSPROP:
            for (int i = 0; i < num_filters; i++) {
                for (int j = 0; j < filter_size; j++) {
                    for (int k = 0; k < filter_size; k++) {
                        for (int l = 0; l < input_channels; l++) {
                            conv_weights[i][j][k][l] -= rmsprop(conv_grads[i][j][k][l], optimizer->optimizer.rmsprop->square_avg_grad[layer_index][counter], optimizer->optimizer.rmsprop->rho, optimizer->optimizer.rmsprop->epsilon, optimizer->optimizer.rmsprop->learning_rate);
                            counter++;
                        }
                    }
                }
                biases[i] -= rmsprop(bias_grads[i], optimizer->optimizer.rmsprop->square_avg_grad[layer_index][num_params + i], optimizer->optimizer.rmsprop->rho, optimizer->optimizer.rmsprop->epsilon, optimizer->optimizer.rmsprop->learning_rate);
            }
            break;
        default:
            fprintf(stderr, "Error: Invalid optimizer type\n");
            exit(1);
    }
}
