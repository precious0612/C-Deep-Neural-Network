/* utils/compute/pooling.c */

#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include "pooling.h"
#include "../memory.h"

float*** pool_forward(float*** input, Dimensions input_shape, Dimensions output_shape, int pool_size, int stride, char* pool_type) {
    // Allocate memory for the output
    float*** output = malloc_3d_float_array(output_shape.height, output_shape.width, output_shape.channels);

    // Iterate over output elements
    for (int out_ch = 0; out_ch < input_shape.channels; out_ch++) {
        for (int out_y = 0; out_y < output_shape.height; out_y++) {
            for (int out_x = 0; out_x < output_shape.width; out_x++) {
                float pool_val = -FLT_MAX;  // Initialize with minimum float value
                // Perform pooling operation
                for (int fy = 0; fy < pool_size; fy++) {
                    for (int fx = 0; fx < pool_size; fx++) {
                        int in_y = out_y * stride + fy;
                        int in_x = out_x * stride + fx;
                        if (in_y < input_shape.height && in_x < input_shape.width) {
                            if (strcmp(pool_type, "max") == 0) {
                                pool_val = fmax(pool_val, input[in_y][in_x][out_ch]);
                            } else if (strcmp(pool_type, "avg") == 0) {
                                pool_val += input[in_y][in_x][out_ch];
                            }
                        }
                    }
                }
                if (strcmp(pool_type, "avg") == 0) {
                    pool_val /= pool_size * pool_size;
                }
                output[out_y][out_x][out_ch] = pool_val;
            }
        }
    }

    return output;
}

void pool_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape, Dimensions output_shape, int pool_size, int stride, char* pool_type) {

    // Compute gradients for the input
    for (int out_ch = 0; out_ch < input_shape.channels; out_ch++) {
        for (int out_y = 0; out_y < output_shape.height; out_y++) {
            for (int out_x = 0; out_x < output_shape.width; out_x++) {
                float pool_grad = output_grad[out_y][out_x][out_ch];
                // Perform pooling gradient operation
                for (int fy = 0; fy < pool_size; fy++) {
                    for (int fx = 0; fx < pool_size; fx++) {
                        int in_y = out_y * stride + fy;
                        int in_x = out_x * stride + fx;
                        if (in_y < input_shape.height && in_x < input_shape.width) {
                            if (strcmp(pool_type, "max") == 0) {
                                float max_val = -FLT_MAX;
                                for (int ky = 0; ky < pool_size; ky++) {
                                    for (int kx = 0; kx < pool_size; kx++) {
                                        int in_ky = out_y * stride + ky;
                                        int in_kx = out_x * stride + kx;
                                        if (in_ky < input_shape.height && in_kx < input_shape.width) {
                                            max_val = fmax(max_val, input[in_ky][in_kx][out_ch]);
                                        }
                                    }
                                }
                                if (input[in_y][in_x][out_ch] == max_val) {
                                    input_grad[in_y][in_x][out_ch] += pool_grad;
                                }
                            } else if (strcmp(pool_type, "avg") == 0) {
                                input_grad[in_y][in_x][out_ch] += pool_grad / (pool_size * pool_size);
                            }
                        }
                    }
                }
            }
        }
    }
}
