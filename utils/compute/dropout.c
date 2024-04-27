/* utils/compute/dropout.c */

#include <stdlib.h>
#include <string.h>

#include "dropout.h"
#include "../rand.h"
#include "../memory.h"
#include "../tensor.h"

float*** dropout_forward(float*** input, Dimensions input_shape, float dropout_rate) {
    // Allocate memory for the output
    float*** output = copy_3d_array(input, input_shape);

    // Apply dropout
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                if (rand_uniform(0.0, 1.0) < dropout_rate) {
                    output[y][x][c] = 0.0f;
                }
            }
        }
    }

    return output;
}

void dropout_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape) {
    // Compute gradients for the input
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                if (input[y][x][c] != 0.0f) {
                    input_grad[y][x][c] = output_grad[y][x][c];
                } else {
                    input_grad[y][x][c] = 0.0f;
                }
            }
        }
    }
}
