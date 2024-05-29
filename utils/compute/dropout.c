//
//  dropout.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/20/24.
//

#include "dropout.h"

#include <stdio.h>

#include "../tools.h"
#include "../memory.h"

float*** dropout_forward(float*** input, int input_width, int input_height, int input_channels, float dropout_rate) {
    
    if (input == NULL) {
        fprintf(stderr, "Dropout Layer input is NULL\n");
        return NULL;
    }
    
    float*** output = copy_3d_float_array(input, input_width, input_height, input_channels);
    if (output == NULL) {
        fprintf(stderr, "Error allocating memory during dropout\n");
        return NULL;
    }
    
    int size = input_width * input_height * input_channels;
    float* output_p = &output[0][0][0];

    // Apply dropout
    for (int i = 0; i < size; ++i) {
        if (rand_uniform(0.0, 1.0) < dropout_rate) {
            output_p[i] = 0.0f;
        }
    }

    return output;
}

void dropout_backward(float*** input, float*** output_grad, float*** input_grad, int input_size) {
    
    if (input == NULL || output_grad == NULL || input_grad == NULL) {
        fprintf(stderr, "Dropout Layer input (backward) is NULL\n");
        return;
    }
    
    float* input_p       = &input[0][0][0];
    float* output_grad_p = &output_grad[0][0][0];
    float* input_grad_p  = &input_grad[0][0][0];
    
    for (int i = 0; i < input_size; ++i) {
        if (input_p[i] != 0.0f) {
            input_grad_p[i] = output_grad_p[i];
        } else {
            input_grad_p[i] = 0.0f;
        }
    }
}
