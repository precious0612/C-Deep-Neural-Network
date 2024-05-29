//
//  losses.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/24/24.
//

#ifndef losses_h
#define losses_h

// MARK: - Define data structures for different loss functions

typedef float (*Loss)(float*, int, int);

typedef enum {
    CrossEntropy,
    MSE
} LossType;

typedef struct {
    LossType type;
    Loss loss_function;
} LossFunction;

// MARK: - Method Declarations

/// Initialize the `LossFunction` struct for specific type.
/// - Parameter type: The `LossType` such as above.
/// - Returns: A pointer of `LossFunction`.
LossFunction* init_loss_function(LossType type);

/// Deallocate the memory of `LossFunction`.
/// - Parameter loss: The pointer of `LossFunction` will be deallocated.
void delete_loss_function(LossFunction* loss);

#endif /* losses_h */
