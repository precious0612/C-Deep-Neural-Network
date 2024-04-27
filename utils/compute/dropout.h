/* utils/compute/dropout.h */

#ifndef DROPOUT_H
#define DROPOUT_H

#include "../dimension.h"

float*** dropout_forward(float*** input, Dimensions input_shape, float dropout_rate);
void dropout_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape);

#endif /* DROPOUT_H */