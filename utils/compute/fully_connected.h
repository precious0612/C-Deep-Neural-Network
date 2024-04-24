/* utils/compute/fully_connected.h */

#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "../dimension.h"
#include "../../optimizer/optimizer.h"

void fc_forward(float*** input, float** output, Dimensions input_shape, Dimensions output_shape, int num_neurons, float** weights, float* biases);
void fc_backward(float*** input, float** output_grad, float*** input_grad, Dimensions input_shape, int num_neurons, float** weights, float* biases, float** weight_grads, float* bias_grads, int prev_layer_is_flatten);
void update_fc_weights(int num_neurons, float** fc_weights, float** fc_grads, float* biases, float* bias_grads, Optimizer* optimizer, int layer_index);

#endif /* FULLY_CONNECTED_H */
