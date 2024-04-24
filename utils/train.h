/* utils/train.h */

#ifndef TRAIN_H
#define TRAIN_H

#include "dimension.h"
#include "loss.h"

void compute_output_grad(float**** batch_outputs, int* batch_labels, LossFunction loss_fn, float**** batch_output_grads, int batch_size, Dimensions output_dim);
int get_prediction(float*** output, const char* metric_name, Dimensions output_dim);

#endif // /* TRAIN_H */