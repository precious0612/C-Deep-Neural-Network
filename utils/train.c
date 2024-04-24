/* utils/train.c */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "train.h"
#include "metric.h"

// Compute output gradient based on loss function
void compute_output_grad(float**** batch_outputs, int* batch_labels, LossFunction loss_fn, float**** batch_output_grads, int batch_size, Dimensions output_dim) {
    int num_classes = output_dim.channels;
    int output_height = output_dim.height;

    for (int i = 0; i < batch_size; i++) {
        int label = batch_labels[i];
        float*** output = batch_outputs[i];
        float*** output_grad = batch_output_grads[i];

        loss_fn(output, label, batch_size, output_height, num_classes, output_grad);
    }
}

// Get prediction based on output and metric
int get_prediction(float*** output, const char* metric_name, Dimensions output_dim) {
    int num_classes = output_dim.width;

    if (strcmp(metric_name, "accuracy") == 0) {
        return get_prediction_accuracy(output, num_classes, output_dim);
    } else if (strcmp(metric_name, "f1_score") == 0) {
        // return compute_f1_score(output, 1, num_classes, output_dim);
    } 
    // Add more metrics as needed
}
