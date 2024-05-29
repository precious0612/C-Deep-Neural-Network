//
//  optimizer.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/18/24.
//

#ifndef optimizer_h
#define optimizer_h

// MARK: - Define data structures for different optimizers

// TODO: Define different types of optimizers
typedef enum {
    SGD,
    ADAM,
    RMSPROP
} OptimizerType;

typedef float LearningRate;

// TODO: Stochastic Gradient Descent (SGD)
typedef struct {
    LearningRate learning_rate;
    float momentum;
    float** momentum_buffer;
} SGDOptimizer;

// TODO: Adam
typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float** m;     // First moment vector
    float** v;     // Second moment vector
    int t;         // Time step
} AdamOptimizer;

// TODO: RMSprop
typedef struct {
    float learning_rate;
    float rho;
    float epsilon;
    float** square_avg_grad; // Moving average of squared gradients
} RMSpropOptimizer;

typedef struct {
    OptimizerType type;
    union {
        SGDOptimizer*     sgd;
        AdamOptimizer*    adam;
        RMSpropOptimizer* rmsprop;
    } optimizer;
} Optimizer;


///  Creates an optimizer based on the specified type.
/// - Parameters:
///   - optimizer_type: The type of optimizer (`SGD`, `Adam`, or `RMSprop`).
///   - learning_rate: The learning rate for the optimizer.
///   - num_weights: An array of each layers' number of weights.
///   - num_layers: The total number of layers with weights in the model.
/// - Returns: A pointer to the initialized Optimizer struct.
///
/// - Example Usage:
///     ```c
///     Optimizer* opt = create_optimizer("Adam", 0.001, num_weights, len(num_weights));
///     ```
///
Optimizer* create_optimizer(OptimizerType optimizer_type, LearningRate learning_rate, int* num_weights, int num_layers);

/// Frees the memory allocated for the optimizer.
/// - Parameters:
///   - optimizer: A pointer to the Optimizer struct to be deleted.
///   - num_layers: The number of layers in the model.
///
/// - Example Usage:
///     ```c
///     delete_optimizer(opt, 5); // Assuming 5 layers in the model
///     ```
///
void delete_optimizer(Optimizer* optimizer, int num_layers);

#endif /* optimizer_h */
