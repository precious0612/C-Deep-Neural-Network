/* utils/tensor.c */

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

// Free memory allocated for a tensor
void free_tensor(float*** tensor, Dimensions tensor_shape) {
    free_3d_float_array(tensor, tensor_shape.height, tensor_shape.width);
}
