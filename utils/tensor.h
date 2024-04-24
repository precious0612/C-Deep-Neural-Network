/* utils/tensor.h */

#ifndef TENSOR_H
#define TENSOR_H

#include "dimension.h"

float*** allocate_output_tensor(Dimensions output_shape);
float*** allocate_grad_tensor(Dimensions shape);
void free_tensor(float*** tensor, Dimensions tensor_shape);

#endif // /* TENSOR_H */