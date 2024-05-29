//
//  activation.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/20/24.
//

#include "activation.h"

#include <math.h>
#include <stdio.h>
#include <float.h>

#include "../memory.h"

static float*** relu_forward(float*** input, int input_width, int input_height, int input_channels) {
    
    float*** output = calloc_3d_float_array(input_width, input_height, input_channels);
    if (output == NULL) {
        fprintf(stderr, "Error allocating memory during activation\n");
        return NULL;
    }
    
    float* input_p  = &input[0][0][0];
    float* output_p = &output[0][0][0];
    int size = input_width * input_height * input_channels;

    for (int i = 0; i < size; ++i) {
        output_p[i] = fmaxf(0.0f, input_p[i]);
    }

    return output;
}

static void relu_backward(float*** output, float*** output_grad, float*** input_grad, int output_size) {
    
    float* output_p      = &output[0][0][0];
    float* output_grad_p = &output_grad[0][0][0];
    float* input_grad_p  = &input_grad[0][0][0];

    for (int i = 0; i < output_size; ++i) {
        input_grad_p[i] = (output_p[i] > 0) ? output_grad_p[i] : 0.0f;
    }
}

static float*** sigmoid_forward(float*** input, int input_width, int input_height, int input_channels) {
    
    float*** output = calloc_3d_float_array(input_width, input_height, input_channels);
    if (output == NULL) {
        fprintf(stderr, "Error allocating memory during activation\n");
        return NULL;
    }
    
    float* input_p  = &input[0][0][0];
    float* output_p = &output[0][0][0];
    int size = input_width * input_height * input_channels;
    
    for (int i = 0; i < size; ++i) {
        output_p[i] = powf(expf(-input_p[i]) + 1.0f, -1.0f);
    }
    
    return output;
}

static void sigmoid_backward(float*** output, float*** output_grad, float*** input_grad, int output_size) {
    
    float* output_p      = &output[0][0][0];
    float* output_grad_p = &output_grad[0][0][0];
    float* input_grad_p  = &input_grad[0][0][0];
    
    for (int i = 0; i < output_size; ++i) {
        input_grad_p[i] = output_grad_p[i] * output_p[i] * (1.0f - output_p[i]);
    }
}

static float*** tanh_forward(float*** input, int input_width, int input_height, int input_channels) {
    
    float*** output = calloc_3d_float_array(input_width, input_height, input_channels);
    if (output == NULL) {
        fprintf(stderr, "Error allocating memory during activation\n");
        return NULL;
    }
    
    float* input_p  = &input[0][0][0];
    float* output_p = &output[0][0][0];
    int size = input_width * input_height * input_channels;
    
    for (int i = 0; i < size; ++i) {
        output_p[i] = tanhf(input_p[i]);
    }
    
    return output;
}

static void tanh_backward(float*** output, float*** output_grad, float*** input_grad, int output_size) {
    
    float* output_p      = &output[0][0][0];
    float* output_grad_p = &output_grad[0][0][0];
    float* input_grad_p  = &input_grad[0][0][0];
    
    for (int i = 0; i < output_size; ++i) {
        input_grad_p[i] = output_grad_p[i] * (1.0f - powf(output_p[i], 2.0f));
    }
}

static float*** softmax_forward(float*** input, int input_width, int input_height, int input_channels) {
    
    if (input == NULL) {
        fprintf(stderr, "ReLU Activation input is NULL\n");
        return NULL;
    }
    
    float*** output = calloc_3d_float_array(input_width, input_height, input_channels);
    if (output == NULL) {
        fprintf(stderr, "Error allocating memory during activation\n");
        return NULL;
    }
    
    float* input_p  = &input[0][0][0];
    float* output_p = &output[0][0][0];
    int size = input_width * input_height * input_channels;
    
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        max_val = fmaxf(max_val, input_p[i]);
    }
    for (int i = 0; i < size; ++i) {
        sum_exp += expf(input_p[i] - max_val);
    }
    for (int i = 0; i < size; ++i) {
        output_p[i] = expf(input_p[i] - max_val) / sum_exp;
    }

    return output;
}

static void softmax_backward(float*** output, float*** output_grad, float*** input_grad, int output_size) {
    
    float* output_p      = &output[0][0][0];
    float* output_grad_p = &output_grad[0][0][0];
    float* input_grad_p  = &input_grad[0][0][0];
    
    float sum_grad = 0.0f;
    
    for (int i = 0; i < output_size; ++i) {
        sum_grad += output_grad_p[i];
    }
    for (int i = 0; i < output_size; ++i) {
        input_grad_p[i] = output_grad_p[i] - sum_grad * output_p[i];
    }
}

float*** forward_activation(const ActivationType activation, float*** input, int input_width, int input_height, int input_channels) {
    
    if (input == NULL) {
        fprintf(stderr, "Activation input is NULL\n");
        return NULL;
    }
    
    float*** output = NULL;
    
    switch (activation) {
        case RELU:
            output = relu_forward(input, input_width, input_height, input_channels);
            break;
            
        case SIGMOID:
            output = sigmoid_forward(input, input_width, input_height, input_channels);
            break;
            
        case SOFTMAX:
            output = softmax_forward(input, input_width, input_height, input_channels);
            break;
    
        default:
            fprintf(stderr, "Error: Unknown activation function\n");
            break;
    }

    return output;
}

void backward_activation(const ActivationType activation, float*** output, float*** output_grad, float*** input_grad, int output_size) {
    
    if (output == NULL || output_grad == NULL || input_grad == NULL) {
        fprintf(stderr, "Activation input (backward) is NULL\n");
        return;
    }
    
    switch (activation) {
        case RELU:
            relu_backward(output, output_grad, input_grad, output_size);
            break;
        
        case SIGMOID:
            sigmoid_backward(output, output_grad, input_grad, output_size);
            break;
            
        case SOFTMAX:
            softmax_backward(output, output_grad, input_grad, output_size);
            break;
            
        default:
            fprintf(stderr, "Error: Unknown activation function\n");
            break;
    }
}
