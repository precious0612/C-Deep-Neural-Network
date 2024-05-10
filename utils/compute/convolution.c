/* utils/compute/convolution.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "convolution.h"
#include "../optim.h"
#include "../memory.h"

#include "arch.h"

float*** conv_forward(float*** input, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float* biases) {
    // Allocate memory for the output
    float*** output = malloc_3d_float_array(output_shape.height, output_shape.width, num_filters);

    // Iterate over output elements
    for (int out_ch = 0; out_ch < num_filters; out_ch++) {
        for (int out_y = 0; out_y < output_shape.height; out_y++) {
            for (int out_x = 0; out_x < output_shape.width; out_x++) {
#ifdef __SSE__
                __m128 sum = _mm_setzero_ps();
#elif __ARM_NEON
                float32x4_t sum = vdupq_n_f32(0.0f);
#else
                float sum = 0.0f;
#endif
                // Perform convolution operation
                for (int in_ch = 0; in_ch < input_shape.channels; in_ch++) {
                    for (int fy = 0; fy < filter_size; fy++) {
                        for (int fx = 0; fx < filter_size; fx++) {
                            int in_y = out_y * stride + fy - padding;
                            int in_x = out_x * stride + fx - padding;
                            if (in_y >= 0 && in_y < input_shape.height && in_x >= 0 && in_x < input_shape.width) {
#ifdef __SSE__
                                __m128 input_val = _mm_load_ss(&input[in_y][in_x][in_ch]);
                                __m128 weight_val = _mm_load_ss(&weights[out_ch][in_ch][fy][fx]);
                                __m128 product = _mm_mul_ps(input_val, weight_val);
                                sum = _mm_add_ps(sum, product);
#elif __ARM_NEON
                                float32x4_t input_val = vld1q_dup_f32(&input[in_y][in_x][in_ch]);
                                float32x4_t weight_val = vld1q_dup_f32(&weights[out_ch][in_ch][fy][fx]);
                                float32x4_t product = vmulq_f32(input_val, weight_val);
                                sum = vaddq_f32(sum, product);
#else
                                sum += input[in_y][in_x][in_ch] * weights[out_ch][in_ch][fy][fx];
#endif
                            }
                        }
                    }
                }
#ifdef __SSE__
                __m128 bias_val = _mm_load_ss(&biases[out_ch]);
                sum = _mm_add_ps(sum, bias_val);
                _mm_store_ss(&output[out_y][out_x][out_ch], sum);
#elif __ARM_NEON
                float32x4_t bias_val = vld1q_dup_f32(&biases[out_ch]);
                sum = vaddq_f32(sum, bias_val);
                vst1q_lane_f32(&output[out_y][out_x][out_ch], sum, 0);
#else
                output[out_y][out_x][out_ch] = sum + biases[out_ch];
#endif
            }
        }
    }

    return output;
}

void conv_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape, Dimensions output_shape, int num_filters, int filter_size, int stride, int padding, float**** weights, float**** weight_grads, float* bias_grads) {
    // Initialize weight and bias gradients to zero
    for (int f = 0; f < num_filters; f++) {
        for (int c = 0; c < input_shape.channels; c++) {
            for (int h = 0; h < filter_size; h++) {
                for (int w = 0; w < filter_size; w++) {
                    weight_grads[f][c][h][w] = 0.0f;
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
#ifdef __SSE__
                                __m128 input_val = _mm_load_ss(&input[in_y][in_x][in_ch]);
                                __m128 weight_val = _mm_load_ss(&weights[out_ch][in_ch][fy][fx]);
                                __m128 output_grad_val = _mm_load_ss(&output_grad[out_y][out_x][out_ch]);
                                __m128 product = _mm_mul_ps(input_val, weight_val);
                                __m128 result = _mm_mul_ps(product, output_grad_val);
                                _mm_store_ss(&input_grad[in_y][in_x][in_ch], _mm_add_ss(_mm_load_ss(&input_grad[in_y][in_x][in_ch]), result));
#elif __ARM_NEON
                                float32x4_t input_val = vld1q_dup_f32(&input[in_y][in_x][in_ch]);
                                float32x4_t weight_val = vld1q_dup_f32(&weights[out_ch][in_ch][fy][fx]);
                                float32x4_t output_grad_val = vld1q_dup_f32(&output_grad[out_y][out_x][out_ch]);
                                float32x4_t product = vmulq_f32(weight_val, output_grad_val);
                                float32x4_t result = vmulq_f32(input_val, product);
                                float32x4_t old_val = vld1q_dup_f32(&input_grad[in_y][in_x][in_ch]);
                                float32x4_t new_val = vaddq_f32(old_val, result);
                                vst1q_lane_f32(&input_grad[in_y][in_x][in_ch], new_val, 0);
#else
                                input_grad[in_y][in_x][in_ch] += output_grad[out_y][out_x][out_ch] * weights[out_ch][in_ch][fy][fx];
#endif                     
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
#ifdef __SSE__
                __m128 output_grad_val = _mm_load_ss(&output_grad[out_y][out_x][out_ch]);
                _mm_store_ss(&bias_grads[out_ch], _mm_add_ss(_mm_load_ss(&bias_grads[out_ch]), output_grad_val));
#elif __ARM_NEON
                float32x4_t output_grad_val = vld1q_dup_f32(&output_grad[out_y][out_x][out_ch]);
                float32x4_t old_val = vld1q_dup_f32(&bias_grads[out_ch]);
                float32x4_t new_val = vaddq_f32(old_val, output_grad_val);
                vst1q_lane_f32(&bias_grads[out_ch], new_val, 0);
#else
                bias_grads[out_ch] += output_grad[out_y][out_x][out_ch];
#endif
                for (int in_ch = 0; in_ch < input_shape.channels; in_ch++) {
                    for (int fy = 0; fy < filter_size; fy++) {
                        for (int fx = 0; fx < filter_size; fx++) {
                            int in_y = out_y * stride + fy - padding;
                            int in_x = out_x * stride + fx - padding;
                            if (in_y >= 0 && in_y < input_shape.height && in_x >= 0 && in_x < input_shape.width) {
#ifdef __SSE__
                                __m128 input_val = _mm_load_ss(&input[in_y][in_x][in_ch]);
                                __m128 output_grad_val = _mm_load_ss(&output_grad[out_y][out_x][out_ch]);
                                __m128 product = _mm_mul_ps(input_val, output_grad_val);
                                _mm_store_ss(&weight_grads[out_ch][in_ch][fy][fx], _mm_add_ss(_mm_load_ss(&weight_grads[out_ch][in_ch][fy][fx]), product));
#elif __ARM_NEON
                                float32x4_t input_val = vld1q_dup_f32(&input[in_y][in_x][in_ch]);
                                float32x4_t output_grad_val = vld1q_dup_f32(&output_grad[out_y][out_x][out_ch]);
                                float32x4_t product = vmulq_f32(input_val, output_grad_val);
                                float32x4_t old_val = vld1q_dup_f32(&weight_grads[out_ch][in_ch][fy][fx]);
                                float32x4_t new_val = vaddq_f32(old_val, product);
                                vst1q_lane_f32(&weight_grads[out_ch][in_ch][fy][fx], new_val, 0);
#else
                                weight_grads[out_ch][in_ch][fy][fx] += input[in_y][in_x][in_ch] * output_grad[out_y][out_x][out_ch];
#endif
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
                for (int j = 0; j < input_channels; j++) {
                    for (int k = 0; k < filter_size; k++) {
                        for (int l = 0; l < filter_size; l++) {
                            conv_weights[i][j][k][l] -= sgd(conv_grads[i][j][k][l], optimizer->optimizer.sgd->momentum, optimizer->optimizer.sgd->momentum_buffer[layer_index][i], optimizer->optimizer.sgd->learning_rate);
                        }
                    }
                }
                biases[i] -= sgd(bias_grads[i], optimizer->optimizer.sgd->momentum, optimizer->optimizer.sgd->momentum_buffer[layer_index][num_params + i], optimizer->optimizer.sgd->learning_rate);
            }
            break;
        case ADAM:
            for (int i = 0; i < num_filters; i++) {
                for (int j = 0; j < input_channels; j++) {
                    for (int k = 0; k < filter_size; k++) {
                        for (int l = 0; l < filter_size; l++) {
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
                for (int j = 0; j < input_channels; j++) {
                    for (int k = 0; k < filter_size; k++) {
                        for (int l = 0; l < filter_size; l++) {
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
