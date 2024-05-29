//
//  pooling.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/20/24.
//

#include "pooling.h"

#include <stdio.h>
#include <float.h>
#include <math.h>

#include "../memory.h"

float*** pool_forward(float*** input, int input_width, int input_height, int channels, int pool_size, int stride, PoolType pool_type) {
    
    if (input == NULL) {
        fprintf(stderr, "Pooling Layer input is NULL\n");
        return NULL;
    }
    
    int output_width  = (input_width - pool_size) / stride + 1;
    int output_height = (input_height - pool_size) / stride + 1;

    float*** output = calloc_3d_float_array(output_width, output_height, channels);
    if (output == NULL) {
        fprintf(stderr, "Allocating memory failed during pooling\n");
        return NULL;
    }

    // Iterate over output elements
    for (int out_ch = 0; out_ch < channels; out_ch++) {
        for (int out_y = 0; out_y < output_width; out_y++) {
            for (int out_x = 0; out_x < output_height; out_x++) {
                float pool_val = -FLT_MAX;
                for (int fy = 0; fy < pool_size; fy++) {
                    for (int fx = 0; fx < pool_size; fx++) {
                        int in_y = out_y * stride + fy;
                        int in_x = out_x * stride + fx;
                        if (in_y < input_width && in_x < input_height) {
                            switch (pool_type) {
                                case MAX:
                                    pool_val = fmaxf(pool_val, input[in_y][in_x][out_ch]);
                                    break;
                                case AVARAGE:
                                    pool_val += input[in_y][in_x][out_ch];
                                    break;
                                default:
                                    fprintf(stderr, "Error: Unknown pool type\n");
                                    break;
                            }
                        }
                    }
                }
                if (pool_type == AVARAGE) {
                    pool_val /= pool_size * pool_size;
                }
                output[out_y][out_x][out_ch] = pool_val;
            }
        }
    }

    return output;
}

void pool_backward(float*** input, float*** output_grad, float*** input_grad, int input_width, int input_height, int channels, int pool_size, int stride, PoolType pool_type) {
    
    if (input == NULL || output_grad == NULL || input_grad == NULL) {
        fprintf(stderr, "Pooling Layer input (backward) is NULL\n");
        return;
    }
    
    int output_width  = (input_width - pool_size) / stride + 1;
    int output_height = (input_height - pool_size) / stride + 1;

    // Compute gradients for the input
    for (int out_ch = 0; out_ch < channels; out_ch++) {
        for (int out_y = 0; out_y < output_width; out_y++) {
            for (int out_x = 0; out_x < output_height; out_x++) {
                float pool_grad = output_grad[out_y][out_x][out_ch];
                // Perform pooling gradient operation
                for (int fy = 0; fy < pool_size; fy++) {
                    for (int fx = 0; fx < pool_size; fx++) {
                        int in_y = out_y * stride + fy;
                        int in_x = out_x * stride + fx;
                        if (in_y < input_width && in_x < input_height) {
                            float max_val = -FLT_MAX;
                            switch (pool_type) {
                                case MAX:
                                    for (int ky = 0; ky < pool_size; ky++) {
                                        for (int kx = 0; kx < pool_size; kx++) {
                                            int in_ky = out_y * stride + ky;
                                            int in_kx = out_x * stride + kx;
                                            if (in_ky < input_width && in_kx < input_height) {
                                                max_val = fmaxf(max_val, input[in_ky][in_kx][out_ch]);
                                            }
                                        }
                                    }
                                    if (input[in_y][in_x][out_ch] == max_val) {
                                        input_grad[in_y][in_x][out_ch] += pool_grad;
                                    }
                                    break;
                                case AVARAGE:
                                    input_grad[in_y][in_x][out_ch] += pool_grad / (pool_size * pool_size);
                                    break;
                                default:
                                    fprintf(stderr, "Error: Unknown pool type\n");
                                    break;
                            }
                        }
                    }
                }
            }
        }
    }
}
