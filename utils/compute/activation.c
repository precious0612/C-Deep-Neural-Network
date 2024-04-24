/* utils/compute/activation.c */

#include <math.h>
#include <string.h>

#include "activation.h"

#define FLT_MAX 3.402823466e+38F

void relu_forward(float*** input, float*** output, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                output[y][x][c] = fmaxf(0, input[y][x][c]);
            }
        }
    }
}

void relu_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                input_grad[y][x][c] = (input[y][x][c] > 0) ? output_grad[y][x][c] : 0;
            }
        }
    }
}

void sigmoid_forward(float*** input, float*** output, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                output[y][x][c] = 1.0f / (1.0f + expf(-input[y][x][c]));
            }
        }
    }
}

void sigmoid_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                float sigmoid_val = input[y][x][c];
                input_grad[y][x][c] = output_grad[y][x][c] * sigmoid_val * (1 - sigmoid_val);
            }
        }
    }
}

void tanh_forward(float*** input, float*** output, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                output[y][x][c] = tanhf(input[y][x][c]);
            }
        }
    }
}

void tanh_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                float tanh_val = input[y][x][c];
                input_grad[y][x][c] = output_grad[y][x][c] * (1 - tanh_val * tanh_val);
            }
        }
    }
}

void max_forward(float*** input, float*** output, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            float max_val = -FLT_MAX;
            for (int c = 0; c < input_shape.channels; c++) {
                max_val = fmaxf(max_val, input[y][x][c]);
            }
            for (int c = 0; c < input_shape.channels; c++) {
                output[y][x][c] = (input[y][x][c] == max_val) ? 1.0f : 0.0f;
            }
        }
    }
}

void max_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            float max_val = -FLT_MAX;
            for (int c = 0; c < input_shape.channels; c++) {
                max_val = fmaxf(max_val, input[y][x][c]);
            }
            for (int c = 0; c < input_shape.channels; c++) {
                input_grad[y][x][c] = (input[y][x][c] == max_val) ? output_grad[y][x][c] : 0.0f;
            }
        }
    }
}

void softmax_forward(float*** input, float*** output, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            float sum_exp = 0.0f;
            for (int c = 0; c < input_shape.channels; c++) {
                sum_exp += expf(input[y][x][c]);
            }
            for (int c = 0; c < input_shape.channels; c++) {
                output[y][x][c] = expf(input[y][x][c]) / sum_exp;
            }
        }
    }
}

void softmax_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            float sum_grad = 0.0f;
            for (int c = 0; c < input_shape.channels; c++) {
                sum_grad += output_grad[y][x][c];
            }
            for (int c = 0; c < input_shape.channels; c++) {
                float softmax_val = input[y][x][c];
                input_grad[y][x][c] = output_grad[y][x][c] - sum_grad * softmax_val;
            }
        }
    }
}

void apply_activation(const char* activation, float*** input, float*** output, Dimensions input_shape, float*** output_grad, float*** input_grad) {
    if (strcmp(activation, "relu") == 0 || strcmp(activation, "ReLU") == 0) {
        relu_forward(input, output, input_shape);
    } else if (strcmp(activation, "sigmoid") == 0) {
        sigmoid_forward(input, output, input_shape);
    } else if (strcmp(activation, "tanh") == 0) {
        tanh_forward(input, output, input_shape);
    } else if (strcmp(activation, "max") == 0) {
        max_forward(input, output, input_shape);
    } else if (strcmp(activation, "softmax") == 0) {
        softmax_forward(input, output, input_shape);
    } else {
        fprintf(stderr, "Error: Unknown activation function %s.\n", activation);
    }
}

void apply_activation_backward(const char* activation, float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape) {
    if (strcmp(activation, "relu") == 0 || strcmp(activation, "ReLU") == 0) {
        relu_backward(input, output_grad, input_grad, input_shape);
    } else if (strcmp(activation, "sigmoid") == 0) {
        sigmoid_backward(input, output_grad, input_grad, input_shape);
    } else if (strcmp(activation, "tanh") == 0) {
        tanh_backward(input, output_grad, input_grad, input_shape);
    } else if (strcmp(activation, "max") == 0) {
        max_backward(input, output_grad, input_grad, input_shape);
    } else if (strcmp(activation, "softmax") == 0) {
        softmax_backward(input, output_grad, input_grad, input_shape);
    } else {
        fprintf(stderr, "Error: Unknown activation function %s.\n", activation);
    }
}
