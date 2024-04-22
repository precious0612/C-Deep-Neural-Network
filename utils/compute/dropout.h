/* utils/compute/dropout.h */

#ifndef DROPOUT_H
#define DROPOUT_H

#include "../dimension.h"

void dropout_forward(float*** input, float*** output, Dimensions input_shape, Dimensions output_shape, float dropout_rate);
void dropout_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape, Dimensions output_shape, float dropout_rate);

#endif /* DROPOUT_H */