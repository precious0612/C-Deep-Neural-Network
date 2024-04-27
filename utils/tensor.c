/* utils/tensor.c */

#include <stdio.h>

#include "tensor.h"
#include "memory.h"

// Allocate memory for output tensor
float*** allocate_output_tensor(Dimensions output_shape) {
    float*** output = malloc_3d_float_array(output_shape.height, output_shape.width, output_shape.channels);
    return output;
}

// Allocate memory for input gradient tensor
float*** allocate_grad_tensor(Dimensions shape) {
    float*** grad = malloc_3d_float_array(shape.height, shape.width, shape.channels);
    return grad;
}

float*** copy_3d_array(float*** src, Dimensions shape) {
    float*** dst = malloc_3d_float_array(shape.height, shape.width, shape.channels);
    if (dst == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for copying 3D array.\n");
        return NULL;
    }

    for (int y = 0; y < shape.height; y++) {
        for (int x = 0; x < shape.width; x++) {
            for (int c = 0; c < shape.channels; c++) {
                dst[y][x][c] = src[y][x][c];
            }
        }
    }

    return dst;
}

// Free memory allocated for a tensor
void free_tensor(float*** tensor, Dimensions tensor_shape) {
    free_3d_float_array(tensor, tensor_shape.height, tensor_shape.width);
}
