//
//  convolution.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/19/24.
//

#include "convolution.h"

#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#include "../memory.h"
#include "optim_algorithm.h"

float*** conv_forward(float*** input, float**** weights, float* biases, int input_width, int input_height, int input_channels, int num_filters, int filter_size, int stride, int padding) {
    
    if (input == NULL || weights == NULL || biases == NULL) {
        fprintf(stderr, "Convolution Layer input is NULL\n");
        return NULL;
    }
    
    int weights_size  = filter_size * filter_size * input_channels;
    int output_width  = (input_width - filter_size + 2 * padding) / stride + 1;
    int output_height = (input_height - filter_size + 2 * padding) / stride + 1;
    int output_size   = output_width * output_height;
    
    float*** output = calloc_3d_float_array(output_width, output_height, num_filters);
    if (output == NULL) {
        fprintf(stderr, "Allocating memory failed during convoluting\n");
        return NULL;
    }
    float* input_paded = (float *)calloc(output_size * weights_size, sizeof(float));
    if (input_paded == NULL) {
        fprintf(stderr, "Allocating memory failed during convoluting\n");
        return NULL;
    }
    
    int index = 0;
    for (int out_col = 0; out_col < output_width; ++out_col) {
        for (int out_row = 0; out_row < output_height; ++out_row) {
            for (int fy = 0; fy < filter_size; ++fy) {
                for (int fx = 0; fx < filter_size; ++fx) {
                    for (int c = 0; c < input_channels; ++c) {
                        int in_col = out_col * stride + fy - padding;
                        int in_row = out_row * stride + fx - padding;
                        if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width) {
                            input_paded[index++] = input[in_col][in_row][c];
                        } else {
                            input_paded[index++] = 0.0f;
                        }
                    }
                }
            }
        }
    }
    
    float* filter_p = &weights[0][0][0][0];  // shape: num_filters * weights_size
    float* output_p = &output[0][0][0];      // shape: output_size * num_filters
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, output_size, num_filters, weights_size, 1.0f, input_paded, weights_size, filter_p, weights_size, 0.0f, output_p, num_filters);
    
    for (index = 0; index < output_size * num_filters; ++index) {
        output_p[index] += biases[index/output_size];
    }
    
    free(input_paded);
    
    return output;
}

void conv_backward(float*** input, float**** weights, float*** output_grad, float*** input_grad, float**** weight_grads, float* bias_grads, int input_width, int input_height, int input_channels, int num_filters, int filter_size, int stride, int padding) {
    
    if (input == NULL || weights == NULL || output_grad == NULL || input_grad == NULL || weight_grads == NULL || bias_grads == NULL) {
        fprintf(stderr, "Convolution Layer input (backward) is NULL\n");
        return;
    }
    
    int weights_size  = filter_size * filter_size * input_channels;
    int output_width  = (input_width - filter_size + 2 * padding) / stride + 1;
    int output_height = (input_height - filter_size + 2 * padding) / stride + 1;
    int output_size   = output_width * output_height;
    
    float* weight_grads_p = &weight_grads[0][0][0][0];
    memset(weight_grads_p, 0, weights_size * num_filters * sizeof(float));
    memset(bias_grads, 0, num_filters * sizeof(float));
    
    float* input_paded   = (float *)calloc(output_size * weights_size, sizeof(float));
    if (input_paded == NULL) {
        fprintf(stderr, "Allocating memory failed during convoluting backward\n");
        return;
    }
    float* output_grad_p = &output_grad[0][0][0];                                    // shape: output_size * num_filters
    float* weights_p     = &weights[0][0][0][0];                                     // shape: num_filters * weights_size
    
    int index = 0;
    for (int out_col = 0; out_col < output_width; ++out_col) {
        for (int out_row = 0; out_row < output_height; ++out_row) {
            for (int fy = 0; fy < filter_size; ++fy) {
                for (int fx = 0; fx < filter_size; ++fx) {
                    for (int c = 0; c < input_channels; ++c) {
                        int in_col = out_col * stride + fy - padding;
                        int in_row = out_row * stride + fx - padding;
                        if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width) {
                            input_paded[index++] = input[in_col][in_row][c];
                        } else {
                            input_paded[index++] = 0.0f;
                        }
                    }
                }
            }
        }
    }
    
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, num_filters, weights_size, output_size, 1.0f, output_grad_p, num_filters, input_paded, weights_size, 0.0f, weight_grads_p, weights_size);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, weights_size, num_filters, 1.0f, output_grad_p, num_filters, weights_p, weights_size, 0.0f, input_paded, weights_size);
    
    float* mask = (float *)malloc(output_size * sizeof(float));
    if (mask == NULL) {
        fprintf(stderr, "Allocating memory failed during convoluting backward\n");
        return;
    }
    for (int i = 0; i < output_size; ++i) {
        mask[i] = 1.0f;
    }
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, num_filters, output_size, 1.0f, mask, output_size, output_grad_p, num_filters, 0.0f, bias_grads, num_filters);
    
    index = 0;
    for (int out_col = 0; out_col < output_width; out_col++) {
        for (int out_row = 0; out_row < output_height; out_row++) {
            for (int fy = 0; fy < filter_size; fy++) {
                for (int fx = 0; fx < filter_size; fx++) {
                    for (int c = 0; c < input_channels; c++) {
                        int in_col = out_col * stride + fy - padding;
                        int in_row = out_row * stride + fx - padding;
                        if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width) {
                            input_grad[in_col][in_row][c] = input_paded[index++];
                        } else {
                            index++;
                        }
                    }
                }
            }
        }
    }
    
    free(input_paded);
    free(mask);
}
