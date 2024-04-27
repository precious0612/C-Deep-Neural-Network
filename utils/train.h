/* utils/train.h
 *
 * This file provides utility functions for training and evaluating deep learning
 * models, including loss computation, gradient computation, prediction generation,
 * and accuracy calculation. These functions are essential for monitoring and
 * optimizing the performance of the model during the training process.
 *
 * Key functionalities include:
 *
 * 1. Computing the loss value for a single output or a batch of outputs, based
 *    on the specified loss function.
 * 2. Computing the gradients of the output with respect to the loss function,
 *    for both single outputs and batches of outputs.
 * 3. Generating predictions from the model's output, based on the specified
 *    evaluation metric (e.g., accuracy, F1-score).
 * 4. Computing the accuracy of the model's predictions for a batch of outputs.
 *
 * This header file serves as a central hub for training and evaluation utilities,
 * providing a consistent and intuitive interface for developers to monitor and
 * optimize their deep learning models.
 *
 * Usage examples:
 *
 * // Compute the loss for a single output
 * float*** output = ...; // Output from the model
 * int true_label = 5; // True label for the input
 * int num_classes = 10; // Number of classes in the output
 * LossFunction loss_fn = categorical_crossentropy_loss; // Specified loss function
 * float loss = compute_loss(output, true_label, num_classes, loss_fn);
 *
 * // Compute the loss for a batch of outputs
 * float**** batch_outputs = ...; // Batch of outputs from the model
 * int* batch_labels = ...; // Batch of true labels
 * int batch_size = 32; // Size of the batch
 * LossFunction loss_fn = categorical_crossentropy_loss; // Specified loss function
 * float batch_loss = compute_loss_batch(batch_outputs, batch_labels, loss_fn, batch_size, num_classes);
 *
 * // Compute the accuracy for a batch of outputs
 * float**** batch_outputs = ...; // Batch of outputs from the model
 * int* batch_labels = ...; // Batch of true labels
 * int batch_size = 32; // Size of the batch
 * int num_classes = 10; // Number of classes in the output
 * float batch_accuracy = compute_accuracy(batch_outputs, batch_labels, batch_size, num_classes);
 */

#ifndef TRAIN_H
#define TRAIN_H

#include "loss.h"

float compute_loss(float*** output, int label, int num_classes, LossFunction loss_fn);
float compute_loss_batch(float**** batch_outputs, int* batch_labels, LossFunction loss_fn, int batch_size, int num_classes);
void compute_output_grad(float*** output, int label, int num_classes, float*** output_grad, int output_height, int output_width);
void compute_output_grad_batch(float**** batch_outputs, int* batch_labels, float**** batch_output_grads, int batch_size, int output_height, int output_width, int num_classes);
int get_prediction(float*** output, const char* metric_name, int num_classes);
float compute_accuracy(float**** batch_outputs, int* batch_labels, int batch_size, int num_classes);

#endif /* TRAIN_H */