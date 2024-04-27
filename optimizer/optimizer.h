/* optimizer/optimizer.h
 *
 * This file defines the interface for different optimization algorithms used to update
 * the weights and biases of the neural network model during the training process.
 *
 * The Optimizer struct serves as a container for various optimizer implementations, including
 * Stochastic Gradient Descent (SGD), Adam, and RMSprop. Each optimizer type is represented
 * by a separate struct, encapsulating the necessary parameters and state variables required
 * for the respective optimization algorithm.
 *
 * Key functionalities include:
 *
 * 1. Initializing different optimizer types (SGD, Adam, RMSprop) with user-specified parameters.
 * 2. Creating an Optimizer instance based on the chosen optimization algorithm.
 * 3. Properly deallocating memory used by the optimizer when it is no longer needed.
 *
 * This header file provides a unified interface for working with various optimization algorithms,
 * enabling developers to easily incorporate different optimization techniques into their neural
 * network training process.
 */

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

/* Define data structures for different optimizers */

// Define different types of optimizers
typedef enum {
    SGD,
    ADAM,
    RMSPROP
} OptimizerType;

/* Stochastic Gradient Descent (SGD) */
typedef struct {
    float learning_rate;
    float momentum;
    float** momentum_buffer;
} SGDOptimizer;

/* Adam */
typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float** m; // First moment vector
    float** v; // Second moment vector
    int t; // Time step
} AdamOptimizer;

/* RMSprop */
typedef struct {
    float learning_rate;
    float rho;
    float epsilon;
    float** square_avg_grad; // Moving average of squared gradients
} RMSpropOptimizer;

typedef struct {
    OptimizerType type;
    union {
        SGDOptimizer* sgd;
        AdamOptimizer* adam;
        RMSpropOptimizer* rmsprop;
    } optimizer;
} Optimizer;

/*
 * Initializes the SGD optimizer with the specified learning rate and momentum.
 *
 * Parameters:
 * - learning_rate: The learning rate for the SGD optimizer.
 * - momentum: The momentum value for the SGD optimizer.
 * - num_weights: An array containing the number of weights for each layer.
 * - num_layers: The number of layers in the model.
 *
 * Returns:
 * - A pointer to the initialized SGDOptimizer struct.
 *
 * Usage example:
 *
 * int num_weights[] = {100, 200, 50};
 * int num_layers = 3;
 * SGDOptimizer* sgd_opt = init_sgd(0.01, 0.9, num_weights, num_layers);
 */
SGDOptimizer* init_sgd(float learning_rate, float momentum, int* num_weights, int num_layers);

/*
 * Initializes the Adam optimizer with the specified parameters.
 *
 * Parameters:
 * - learning_rate: The learning rate for the Adam optimizer.
 * - beta1: The value of the exponential decay rate for the first moment estimates.
 * - beta2: The value of the exponential decay rate for the second-moment estimates.
 * - epsilon: A small constant for numerical stability.
 * - num_weights: An array containing the number of weights for each layer.
 * - num_layers: The number of layers in the model.
 *
 * Returns:
 * - A pointer to the initialized AdamOptimizer struct.
 *
 * Usage example:
 *
 * int num_weights[] = {100, 200, 50};
 * int num_layers = 3;
 * AdamOptimizer* adam_opt = init_adam(0.001, 0.9, 0.999, 1e-8, num_weights, num_layers);
 */
AdamOptimizer* init_adam(float learning_rate, float beta1, float beta2, float epsilon, int* num_weights, int num_layers);

/*
 * Initializes the RMSprop optimizer with the specified parameters.
 *
 * Parameters:
 * - learning_rate: The learning rate for the RMSprop optimizer.
 * - rho: The value of the exponential decay rate for the moving averages.
 * - epsilon: A small constant for numerical stability.
 * - num_weights: An array containing the number of weights for each layer.
 * - num_layers: The number of layers in the model.
 *
 * Returns:
 * - A pointer to the initialized RMSpropOptimizer struct.
 *
 * Usage example:
 *
 * int num_weights[] = {100, 200, 50};
 * int num_layers = 3;
 * RMSpropOptimizer* rmsprop_opt = init_rmsprop(0.001, 0.9, 1e-8, num_weights, num_layers);
 */
RMSpropOptimizer* init_rmsprop(float learning_rate, float rho, float epsilon, int* num_weights, int num_layers);

/*
 * Creates an optimizer based on the specified type.
 *
 * Parameters:
 * - optimizer_type: A string representing the type of optimizer ("SGD", "Adam", or "RMSprop").
 * - learning_rate: The learning rate for the optimizer.
 * - num: The total number of weights in the model.
 *
 * Returns:
 * - A pointer to the initialized Optimizer struct.
 *
 * Usage example:
 *
 * Optimizer* opt = create_optimizer("Adam", 0.001, 1000);
 */
Optimizer* create_optimizer(char* optimizer_type, float learning_rate, int num);

/*
 * Frees the memory allocated for the optimizer.
 *
 * Parameters:
 * - optimizer: A pointer to the Optimizer struct to be deleted.
 * - num_layers: The number of layers in the model.
 *
 * Usage example:
 *
 * Optimizer* opt = create_optimizer("SGD", 0.01, 1000);
 * // Use the optimizer for training
 * ...
 * delete_optimizer(opt, 5); // Assuming 5 layers in the model
 */
void delete_optimizer(Optimizer* optimizer, int num_layers);

#endif /* OPTIMIZER_H */