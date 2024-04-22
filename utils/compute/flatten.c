/* utils/compute/flatten.c */

#include "flatten.h"

void flatten_input(float*** input, float* flattened, Dimensions input_shape) {
    int index = 0;
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                flattened[index++] = input[y][x][c];
            }
        }
    }
}
