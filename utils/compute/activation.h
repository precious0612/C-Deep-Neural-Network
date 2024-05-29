//
//  activation.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/20/24.
//

#ifndef activation_h
#define activation_h

typedef enum {
    RELU,
    SIGMOID,
    TANH,
    SOFTMAX,
    NONE
} ActivationType;

float*** forward_activation(const ActivationType activation, float*** input, int input_width, int input_height, int input_channels);

void backward_activation(const ActivationType activation, float*** output, float*** output_grad, float*** input_grad, int output_size);

#endif /* activation_h */
