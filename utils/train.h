/* utils/train.h */

#ifndef TRAIN_H
#define TRAIN_H

#include "loss.h"

float compute_loss(float*** output, int label, int num_classes, LossFunction loss_fn);
float compute_loss_batch(float**** batch_outputs, int* batch_labels, LossFunction loss_fn, int batch_size, int num_classes);
void compute_output_grad(float*** output, int label, int num_classes, float*** output_grad, int output_height, int output_width);
void compute_output_grad_batch(float**** batch_outputs, int* batch_labels, float**** batch_output_grads, int batch_size, int output_height, int output_width, int num_classes);
int get_prediction(float*** output, const char* metric_name, int num_classes);
float compute_accuracy(float**** batch_outputs, int* batch_labels, int batch_size, int num_classes);

#endif // /* TRAIN_H */