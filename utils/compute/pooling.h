/* utils/compute/pooling.h */

#ifndef POOLING_H
#define POOLING_H

#include "../dimension.h"

void pool_forward(float*** input, float*** output, Dimensions input_shape, Dimensions output_shape, int pool_size, int stride, char* pool_type);
void pool_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape, Dimensions output_shape, int pool_size, int stride, char* pool_type);

#endif /* POOLING_H */
