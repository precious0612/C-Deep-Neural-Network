/* optimizer/optimizer.h */

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

// Initialize optimizer
SGDOptimizer* init_sgd(float learning_rate, float momentum, int* num_weights, int num_layers);
AdamOptimizer* init_adam(float learning_rate, float beta1, float beta2, float epsilon, int* num_weights, int num_layers);
RMSpropOptimizer* init_rmsprop(float learning_rate, float rho, float epsilon, int* num_weights, int num_layers);

// Free memory allocated for optimizer
void delete_optimizer(Optimizer* optimizer, int num_layers);

#endif /* OPTIMIZER_H */