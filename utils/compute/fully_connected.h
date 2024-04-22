/* utils/compute/fully_connected.h */

#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "../dimension.h"

void fc_forward(float*** input, float** output, Dimensions input_shape, Dimensions output_shape, int num_neurons, float** weights, float* biases);
void fc_backward(float*** input, float** output_grad, float*** input_grad, Dimensions input_shape, int num_neurons, float** weights, float* biases, float learning_rate, int prev_layer_is_flatten);

#endif /* FULLY_CONNECTED_H */
