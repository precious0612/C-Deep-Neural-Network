//
//  fully_connected.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/21/24.
//

#ifndef fully_connected_h
#define fully_connected_h

float*** fc_forward(float*** input, int channels, int num_neurons, float** weights, float* biases);

void fc_backward(float*** input, float*** output_grad, float*** input_grad, float** weights, float**weight_grads, float* biases, float* bias_grads, int channels, int num_neurons);

#endif /* fully_connected_h */
