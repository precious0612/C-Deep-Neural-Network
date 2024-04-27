/* utils/train.c */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "train.h"
#include "metric.h"

// Compute loss based on loss function
float compute_loss(float*** output, int label, int num_classes, LossFunction loss_fn) {
    return loss_fn(output[0][0], label, num_classes);
}

// Compute loss based on loss function for a batch of outputs
float compute_loss_batch(float**** batch_outputs, int* batch_labels, LossFunction loss_fn, int batch_size, int num_classes) {

    float total_loss = 0.0f;

    for (int i = 0; i < batch_size; i++) {
        int label = batch_labels[i];
        float* output = batch_outputs[i][0][0]; // Assuming flattened output tensor

        total_loss += loss_fn(output, label, num_classes);
    }

    return total_loss / batch_size;
}

// Compute output gradient based on loss function
void compute_output_grad(float*** output, int label, int num_classes, float*** output_grad, int output_height, int output_width) {
    compute_output_grad_batch(&output, &label, &output_grad, 1, output_height, output_width, num_classes);
}

// Compute output gradient based on loss function for a batch of outputs
void compute_output_grad_batch(float**** batch_outputs, int* batch_labels, float**** batch_output_grads, int batch_size, int output_height, int output_width, int num_classes) {

    for (int i = 0; i < batch_size; i++) {
        int label = batch_labels[i];
        float* output = batch_outputs[i][0][0]; // Assuming flattened output tensor

        for (int j = 0; j < output_height * output_width * num_classes; j++) {
            if (j == label) {
                batch_output_grads[i][0][0][j] = output[j] - 1.0f;
            } else {
                batch_output_grads[i][0][0][j] = output[j];
            }
        }
    }
}

// Get prediction based on output and metric
int get_prediction(float*** output, const char* metric_name, int num_classes) {

    if (strcmp(metric_name, "accuracy") == 0) {
        return get_prediction_accuracy(output[0][0], num_classes);
    } else if (strcmp(metric_name, "f1_score") == 0) {
        // return compute_f1_score(output, 1, num_classes);
    } 
    // Add more metrics as needed
    return -1;
}

// Compute accuracy of the model
float compute_accuracy(float**** batch_outputs, int* batch_labels, int batch_size, int num_classes) {
    int correct_predictions = 0;

    for (int i = 0; i < batch_size; i++) {
        int label = batch_labels[i];
        float*** output = batch_outputs[i]; // Assuming flattened output tensor

        if (get_prediction(output, "accuracy", num_classes) == label) {
            correct_predictions++;
        }
    }

    return (float) correct_predictions / batch_size;
}
