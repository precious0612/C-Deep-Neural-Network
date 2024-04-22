/* utils/compute/fully_connected.c */ 

#include <stdlib.h>
#include <string.h>

#include "fully_connected.h"
#include "../memory.h"
#include "flatten.h"

void fc_forward(float*** input, float** output, Dimensions input_shape, Dimensions output_shape, int num_neurons, float** weights, float* biases) {
    int input_size = input_shape.height * input_shape.width * input_shape.channels;

    // Flatten input
    float* flattened_input = malloc_1d_float_array(input_size);
    flatten_input(input, flattened_input, input_shape);

    // Compute output
    for (int n = 0; n < num_neurons; n++) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += flattened_input[i] * weights[n][i];
        }
        output[0][n] = sum + biases[n];
    }

    free(flattened_input);
}

void fc_backward(float*** input, float** output_grad, float*** input_grad, Dimensions input_shape, int num_neurons, float** weights, float* biases, float learning_rate, int prev_layer_is_flatten) {
    int input_size = input_shape.height * input_shape.width * input_shape.channels;

    // Flatten input
    float* flattened_input = malloc_1d_float_array(input_size);
    flatten_input(input, flattened_input, input_shape);

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
                    input_grad[i][j][k] += output_grad[0][n] * weights[n][i];
                    }
                }
            }
        }
    }

    // Compute gradients for the weights and biases
    float** weight_grad = malloc_2d_float_array(num_neurons, input_size);
    float* bias_grad = calloc(num_neurons, sizeof(float));

    for (int n = 0; n < num_neurons; n++) {
        bias_grad[n] = output_grad[0][n];
        for (int i = 0; i < input_size; i++) {
            weight_grad[n][i] = flattened_input[i] * output_grad[0][n];
        }
    }

    // Update weights and biases
    for (int n = 0; n < num_neurons; n++) {
        for (int i = 0; i < input_size; i++) {
            weights[n][i] -= learning_rate * weight_grad[n][i];
        }
        biases[n] -= learning_rate * bias_grad[n];
    }

    free(flattened_input);
    free_2d_float_array(weight_grad, num_neurons);
    free(bias_grad);
}
