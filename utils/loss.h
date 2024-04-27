/* utils/loss.h */

#ifndef LOSS_H
#define LOSS_H

/*
 *This `LossFunction` type is a pointer to a function that takes the following arguments:
 * - `float***`: The output tensor
 * - `int`: The true label
 * - `int`: The batch size
 * - `int`: The height of the output tensor
 * - `int`: The number of classes (channels) in the output tensor
 * - `float***`: The output gradient tensor to be filled by the function
 */
typedef float (*LossFunction)(float*, int, int);
float categorical_crossentropy_loss(float* output, int label, int num_classes);
float mean_squared_error_loss(float* output, int label, int num_classes);

#endif // /* LOSS_H */