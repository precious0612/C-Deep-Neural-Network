//
//  fully_connected.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/21/24.
//

#include "fully_connected.h"

#include <cblas.h>
#include <string.h>
#include <stdlib.h>

#include "../memory.h"

float*** fc_forward(float*** input, int channels, int num_neurons, float** weights, float* biases) {
    
    if (input == NULL || weights == NULL || biases == NULL) {
        fprintf(stderr, "Fully-connected Layer input is NULL\n");
        return NULL;
    }
    
    float*** output = calloc_3d_float_array(1, 1, num_neurons);
    if (output == NULL) {
        fprintf(stderr, "Allocating memory failed during Fully-connecting\n");
        return NULL;
    }
    
    float* input_p   = &input[0][0][0];
    float* output_p  = &output[0][0][0];
    float* weights_p = &weights[0][0];
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, num_neurons, channels, 1.0f, input_p, channels, weights_p, channels, 0.0f, output_p, num_neurons);
    
    return output;
}

void fc_backward(float*** input, float*** output_grad, float*** input_grad, float** weights, float**weight_grads, float* biases, float* bias_grads, int channels, int num_neurons) {
    
    if (input == NULL || weights == NULL || output_grad == NULL || input_grad == NULL || weight_grads == NULL || bias_grads == NULL) {
        fprintf(stderr, "Fully-connected Layer input (backward) is NULL\n");
        return;
    }
    
    float* weight_grads_p = &weight_grads[0][0];
    memset(weight_grads_p, 0, num_neurons * channels * sizeof(float));
    memset(bias_grads, 0, num_neurons * sizeof(float));
    
    float* input_p       = &input[0][0][0];
    float* input_grad_p  = &input_grad[0][0][0];
    float* output_grad_p = &output_grad[0][0][0];
    float* weights_p     = &weights[0][0];
    
    float* temp = (float *)calloc(channels, sizeof(float));
    if (temp == NULL) {
        fprintf(stderr, "Allocating memory failed during Fully-connecting backward\n");
        return;
    }
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, channels, num_neurons, 1.0f, output_grad_p, num_neurons, weights_p, channels, 0.0f, temp, channels);
    cblas_saxpby(channels, 1.0f, temp, 1, 1.0f, input_grad_p, 1);
    
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, num_neurons, channels, 1, 1.0f, output_grad_p, num_neurons, input_p, channels, 0.0f, weight_grads_p, channels);
    
    free(temp);
    temp = NULL;
}
