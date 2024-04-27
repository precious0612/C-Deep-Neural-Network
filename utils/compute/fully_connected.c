/* utils/compute/fully_connected.c */ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fully_connected.h"
#include "../memory.h"
#include "../optim.h"
#include "flatten.h"

float* fc_forward(float*** input, Dimensions input_shape, int num_neurons, float** weights, float* biases) {
    // Allocate memory for the output
    float* output = malloc_1d_float_array(num_neurons);
    int input_size = input_shape.height * input_shape.width * input_shape.channels;

    // Flatten input
    float* flattened_input = malloc_1d_float_array(input_size);
    if (input_shape.height == 1 && input_shape.width == 1) {
        flattened_input = input[0][0];
    } else {
        flatten(input, flattened_input, input_shape);
    }

    // Compute output
    for (int n = 0; n < num_neurons; n++) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += flattened_input[i] * weights[n][i];
        }
        output[n] = sum + biases[n];
    }

    free(flattened_input);
    return output;
}

void fc_backward(float*** input, float** output_grad, float*** input_grad, Dimensions input_shape, int num_neurons, float** weights, float** weight_grads, float* bias_grads, int prev_layer_is_flatten) {
    int input_size = input_shape.height * input_shape.width * input_shape.channels;

    // Initialize weight and bias gradients to zero
    for (int n = 0; n < num_neurons; n++) {
        bias_grads[n] = 0.0f;
        for (int i = 0; i < input_size; i++) {
            weight_grads[n][i] = 0.0f;
        }
    }

    // Flatten input
    float* flattened_input = malloc_1d_float_array(input_size);
    flatten(input, flattened_input, input_shape);

    // Compute gradients for the input
    if (prev_layer_is_flatten) {
        for (int i = 0; i < input_size; i++) {
            for (int n = 0; n < num_neurons; n++) {
                input_grad[0][0][i] += output_grad[0][n] * weights[n][i];
            }
        }
    } else {
        for (int i = 0; i < input_shape.height; i++) {
            for (int j = 0; j < input_shape.width; j++) {
                for (int k = 0; k < input_shape.channels; k++) {
                    for (int n = 0; n < num_neurons; n++) {
                        input_grad[i][j][k] += output_grad[0][n] * weights[n][i * input_shape.width * input_shape.channels + j * input_shape.channels + k];
                    }
                }
            }
        }
    }

    // Compute gradients for the weights and biases
    for (int n = 0; n < num_neurons; n++) {
        bias_grads[n] = output_grad[0][n];
        for (int i = 0; i < input_size; i++) {
            weight_grads[n][i] = flattened_input[i] * output_grad[0][n];
        }
    }

    free(flattened_input);
}

void update_fc_weights(int num_neurons, float** fc_weights, float** fc_grads, float* biases, float* bias_grads, Optimizer* optimizer, int layer_index) {
    switch (optimizer->type) {
        case SGD:
            for (int i = 0; i < num_neurons; i++) {
                fc_weights[layer_index][i] -= sgd(fc_grads[layer_index][i], optimizer->optimizer.sgd->momentum, optimizer->optimizer.sgd->momentum_buffer[layer_index][i], optimizer->optimizer.sgd->learning_rate);
                biases[i] -= sgd(bias_grads[i], optimizer->optimizer.sgd->momentum, optimizer->optimizer.sgd->momentum_buffer[layer_index][num_neurons + i], optimizer->optimizer.sgd->learning_rate);
            }
            break;
        case ADAM:
            for (int i = 0; i < num_neurons; i++) {
                fc_weights[layer_index][i] -= adam(fc_grads[layer_index][i], optimizer->optimizer.adam->t, optimizer->optimizer.adam->m[layer_index][i], optimizer->optimizer.adam->v[layer_index][i], optimizer->optimizer.adam->beta1, optimizer->optimizer.adam->beta2, optimizer->optimizer.adam->epsilon, optimizer->optimizer.adam->learning_rate);
                biases[i] -= adam(bias_grads[i], optimizer->optimizer.adam->t, optimizer->optimizer.adam->m[layer_index][num_neurons + i], optimizer->optimizer.adam->v[layer_index][num_neurons + i], optimizer->optimizer.adam->beta1, optimizer->optimizer.adam->beta2, optimizer->optimizer.adam->epsilon, optimizer->optimizer.adam->learning_rate);
            }
            optimizer->optimizer.adam->t++;
            break;
        case RMSPROP:
            for (int i = 0; i < num_neurons; i++) {
                fc_weights[layer_index][i] -= rmsprop(fc_grads[layer_index][i], optimizer->optimizer.rmsprop->square_avg_grad[layer_index][i], optimizer->optimizer.rmsprop->rho, optimizer->optimizer.rmsprop->epsilon, optimizer->optimizer.rmsprop->learning_rate);
                biases[i] -= rmsprop(bias_grads[i], optimizer->optimizer.rmsprop->square_avg_grad[layer_index][num_neurons + i], optimizer->optimizer.rmsprop->rho, optimizer->optimizer.rmsprop->epsilon, optimizer->optimizer.rmsprop->learning_rate);
            }
            break;
        default:
            fprintf(stderr, "Error: Invalid optimizer type\n");
            exit(1);
    }
}
