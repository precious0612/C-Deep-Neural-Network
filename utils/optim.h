/* utils/optim.h
 *
 * This file provides implementations for various optimization algorithms commonly
 * used in deep learning to update the weights and biases of neural networks during
 * the training process. Optimization algorithms aim to minimize the loss function
 * by adjusting the model's parameters in the direction of the negative gradient.
 *
 * Key functionalities include:
 *
 * 1. Implementing the Stochastic Gradient Descent (SGD) optimization algorithm
 *    with momentum.
 * 2. Implementing the Adam optimization algorithm, which combines momentum with
 *    adaptive learning rates.
 * 3. Implementing the RMSprop optimization algorithm, which uses a moving average
 *    of squared gradients to adaptively adjust the learning rates.
 *
 * This header file serves as a central hub for working with optimization algorithms,
 * ensuring consistency and simplifying the integration of different optimization
 * techniques into deep learning models.
 *
 * Usage examples:
 *
 * // Perform a weight update using SGD with momentum
 * float grads = ...; // Computed gradients for a weight
 * float momentum = 0.9; // Momentum hyperparameter
 * float momentum_buffer = 0.0; // Initialize momentum buffer to zero
 * float learning_rate = 0.001; // Learning rate
 * float weight_update = sgd(grads, momentum, momentum_buffer, learning_rate);
 *
 * // Perform a weight update using Adam
 * float grads = ...; // Computed gradients for a weight
 * int t = 10; // Current time step
 * float m = 0.0; // Initialize first moment vector to zero
 * float v = 0.0; // Initialize second moment vector to zero
 * float beta1 = 0.9; // Exponential decay rate for the first moment estimates
 * float beta2 = 0.999; // Exponential decay rate for the second-moment estimates
 * float epsilon = 1e-8; // Small constant for numerical stability
 * float learning_rate = 0.001; // Learning rate
 * float weight_update = adam(grads, t, m, v, beta1, beta2, epsilon, learning_rate);
 *
 * // Perform a weight update using RMSprop
 * float grads = ...; // Computed gradients for a weight
 * float square_avg_grad = 0.0; // Initialize the moving average of squared gradients to zero
 * float rho = 0.9; // Exponential decay rate for the moving average
 * float epsilon = 1e-8; // Small constant for numerical stability
 * float learning_rate = 0.001; // Learning rate
 * float weight_update = rmsprop(grads, square_avg_grad, rho, epsilon, learning_rate);
 */

#ifndef OPTIM_H
#define OPTIM_H

float sgd(float grads, float momentum, float momentum_buffer, float learning_rate);
float adam(float grads, int t, float m, float v, float beta1, float beta2, float epsilon, float learning_rate);
float rmsprop(float grads, float square_avg_grad, float rho, float epsilon, float learning_rate);

#endif /* OPTIM_H */