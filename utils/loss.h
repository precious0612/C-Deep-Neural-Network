/* utils/loss.h
 *
 * This file provides implementations for various loss functions commonly used in
 * deep learning applications, such as classification and regression tasks. Loss
 * functions measure the discrepancy between the predicted output of a model and
 * the ground truth labels, enabling the model to learn and improve its performance.
 *
 * The `LossFunction` type is defined as a pointer to a function that takes the
 * predicted output, the true label, and the number of classes as input, and returns
 * the computed loss value. This flexible design allows for easy integration of
 * different loss functions into the codebase.
 *
 * Key functionalities include:
 *
 * 1. Defining the `LossFunction` type as a function pointer.
 * 2. Providing implementations for common loss functions, such as Categorical
 *    Cross-Entropy Loss and Mean Squared Error Loss.
 * 3. Enabling the use of custom loss functions by allowing developers to implement
 *    and integrate their own loss functions following the `LossFunction` signature.
 *
 * This header file serves as a central hub for working with loss functions, ensuring
 * consistency and simplifying the integration of different loss functions into deep
 * learning models.
 *
 * Usage examples:
 *
 * // Using Categorical Cross-Entropy Loss
 * float* output = ...; // Predicted output from the model
 * int true_label = 3; // True label for the input
 * int num_classes = 10; // Number of classes in the output
 * float loss = categorical_crossentropy_loss(output, true_label, num_classes);
 *
 * // Using Mean Squared Error Loss
 * float* output = ...; // Predicted output from the model
 * int true_label = 5; // True label for the input
 * int num_classes = 1; // Number of classes in the output (for regression tasks)
 * float loss = mean_squared_error_loss(output, true_label, num_classes);
 */

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