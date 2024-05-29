//
//  flatten.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/21/24.
//

#include "flatten.h"

#include <string.h>
#include <stdio.h>

#include "../memory.h"

float*** flatten(float*** input, int input_width, int input_height, int input_channles) {
    
    if (input == NULL) {
        fprintf(stderr, "Flatten Layer input is NULL\n");
        return NULL;
    }
    
    int size = input_width * input_height * input_channles;
    
    float*** output = calloc_3d_float_array(1, 1, size);
    if (output == NULL) {
        fprintf(stderr, "Allocating memory failed during Flatting\n");
        return NULL;
    }
    
    float* input_p  = &input[0][0][0];
    float* output_p = &output[0][0][0];
    memcpy(output_p, input_p, size * sizeof(float));
    
    return output;
}

float*** unflatten(float*** input, int output_width, int output_height, int output_channles) {
    
    if (input == NULL) {
        fprintf(stderr, "Flatten Layer input is NULL\n");
        return NULL;
    }
    
    int size = output_width * output_height * output_channles;
    
    float*** output = calloc_3d_float_array(output_width, output_height, output_channles);
    if (output == NULL) {
        fprintf(stderr, "Allocating memory failed during Flatting\n");
        return NULL;
    }
    
    float* input_p  = &input[0][0][0];
    float* output_p = &output[0][0][0];
    memcpy(output_p, input_p, size * sizeof(float));
    
    return output;
}
