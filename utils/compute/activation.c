/* utils/compute/activation.c */

#include <math.h>
#include <string.h>
#include <stdio.h>

#include "activation.h"
#include "../memory.h"

// #define FLT_MAX 3.402823466e+38F

float*** relu_forward(float*** input, Dimensions input_shape) {
    float*** output = malloc_3d_float_array(input_shape.height, input_shape.width, input_shape.channels);

    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                output[y][x][c] = fmaxf(0, input[y][x][c]);
            }
        }
    }

    return output;
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

float*** sigmoid_forward(float*** input, Dimensions input_shape) {
    float*** output = malloc_3d_float_array(input_shape.height, input_shape.width, input_shape.channels);

    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                output[y][x][c] = 1.0f / (1.0f + expf(-input[y][x][c]));
            }
        }
    }

    return output;
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

float*** tanh_forward(float*** input, Dimensions input_shape) {
    float*** output = malloc_3d_float_array(input_shape.height, input_shape.width, input_shape.channels);

    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            for (int c = 0; c < input_shape.channels; c++) {
                output[y][x][c] = tanhf(input[y][x][c]);
            }
        }
    }

    return output;
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

float*** max_forward(float*** input, Dimensions input_shape) {
    float*** output = malloc_3d_float_array(input_shape.height, input_shape.width, input_shape.channels);

    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            float max_val = -FLT_MAX;
            int max_idx = 0;
            for (int c = 0; c < input_shape.channels; c++) {
                if (input[y][x][c] > max_val) {
                    max_val = input[y][x][c];
                    max_idx = c;
                }
            }
            for (int c = 0; c < input_shape.channels; c++) {
                output[y][x][c] = (c == max_idx) ? 1.0f : 0.0f;
            }
        }
    }

    return output;
}

void max_backward(float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape) {
    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            float max_val = -FLT_MAX;
            int max_idx = 0;
            for (int c = 0; c < input_shape.channels; c++) {
                if (input[y][x][c] > max_val) {
                    max_val = input[y][x][c];
                    max_idx = c;
                }
            }
            for (int c = 0; c < input_shape.channels; c++) {
                input_grad[y][x][c] = (c == max_idx) ? output_grad[y][x][c] : 0.0f;
            }
        }
    }
}

float*** softmax_forward(float*** input, Dimensions input_shape) {
    float*** output = malloc_3d_float_array(input_shape.height, input_shape.width, input_shape.channels);

    for (int y = 0; y < input_shape.height; y++) {
        for (int x = 0; x < input_shape.width; x++) {
            float max_val = -FLT_MAX;
            for (int c = 0; c < input_shape.channels; c++) {
                max_val = fmaxf(max_val, input[y][x][c]);
            }
            float sum_exp = 0.0f;
            for (int c = 0; c < input_shape.channels; c++) {
                sum_exp += expf(input[y][x][c] - max_val);
            }
            for (int c = 0; c < input_shape.channels; c++) {
                output[y][x][c] = expf(input[y][x][c] - max_val) / sum_exp;
            }
        }
    }

    return output;
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

float*** forward_activation(const char* activation, float*** input, Dimensions input_shape) {
    float*** output = NULL;
    if (strcmp(activation, "relu") == 0 || strcmp(activation, "ReLU") == 0) {
        output = relu_forward(input, input_shape);
    } else if (strcmp(activation, "sigmoid") == 0) {
        output = sigmoid_forward(input, input_shape);
    } else if (strcmp(activation, "tanh") == 0) {
        output = tanh_forward(input, input_shape);
    } else if (strcmp(activation, "max") == 0) {
        output = max_forward(input, input_shape);
    } else if (strcmp(activation, "softmax") == 0) {
        output = softmax_forward(input, input_shape);
    } else {
        fprintf(stderr, "Error: Unknown activation function %s.\n", activation);
    }
    return output;
}

void backward_activation(const char* activation, float*** input, float*** output_grad, float*** input_grad, Dimensions input_shape) {
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
