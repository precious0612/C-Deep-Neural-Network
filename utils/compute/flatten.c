/* utils/compute/flatten.c */

#include "flatten.h"

void flatten(float*** input, float* flattened, Dimensions input_shape) {
    int index = 0;
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                flattened[index++] = input[y][x][c];
            }
        }
    }
}

void unflatten(float* flattened, float*** output, Dimensions output_shape) {
    int index = 0;
    for (int y = 0; y < output_shape.height; y++) {
        for (int x = 0; x < output_shape.width; x++) {
            for (int c = 0; c < output_shape.channels; c++) {
                output[y][x][c] = flattened[index++];
            }
        }
    }
}

