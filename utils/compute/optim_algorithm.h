//
//  optim_algorithm.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/20/24.
//

#ifndef optim_algorithm_h
#define optim_algorithm_h

/// Updating the weights and biases with the weight gradients by using SGD method.
/// - Parameters:
///   - weights: The Layer weights storing in 1D array.
///   - weight_grads: The Layer gradients of weights storing in 1D array.
///   - momentum: The momentum parameter. (Normally is 0.9)
///   - momentum_buffer: The momentum for the weights.
///   - learning_rate: The ratio of updating.
///   - length: The number of weights.
void sgd(float* weights, float* weight_grads, float momentum, float* momentum_buffer, float learning_rate, int length);

/// Updating the weights and biases with the weight gradients by using Adam method.
/// - Parameters:
///   - weights: The Layer weights storing in 1D array.
///   - weight_grads: The Layer gradients of weights storing in 1D array.
///   - m: The first moment estimate.
///   - v: The second raw moment estimate.
///   - beta1: The exponential decay rate of the first-order moment estimate (e.g., 0.9).
///   - beta2: The exponential decay rate (e.g., 0.999) of the second-order moment estimate.
///   - epsilon: This parameter is a very small number, its to prevent dividing by zero in the implementation (e.g. 10E-8).
///   - t: The time of updating.
///   - learning_rate: The ratio of updating.
///   - length: The number of weights.
void adam(float* weights, float* weight_grads, float* m, float* v, float beta1, float beta2, float epsilon, int t, float learning_rate, int length);

/// Updating the weights and biases with the weight gradients by using RMSprop method.
/// - Parameters:
///   - weights: The Layer weights storing in 1D array.
///   - weight_grads: The Layer gradients of weights storing in 1D array.
///   - square_avg_grad: The gradient is weighted average.
///   - rho: It's a decay rate. (Normally is 0.9)
///   - epsilon: This parameter is a very small number, its to prevent dividing by zero in the implementation (e.g. 10E-8).
///   - learning_rate: The ratio of updating.
///   - length: The number of weights.
void rmsprop(float* weights, float* weight_grads, float* square_avg_grad, float rho, float epsilon, float learning_rate, int length);

#endif /* optim_algorithm_h */
