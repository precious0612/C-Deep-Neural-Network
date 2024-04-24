/* utils/compute/flatten.h */

#ifndef FLATTEN_H
#define FLATTEN_H

#include "../dimension.h"

void flatten(float*** input, float* flattened, Dimensions input_shape);
void unflatten(float* flattened, float*** output, Dimensions output_shape);

#endif // /* FLATTEN_ */