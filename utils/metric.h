/* utils/metric.h
 *
 * This file provides implementations for various evaluation metrics commonly used
 * to assess the performance of machine learning models, particularly in classification
 * tasks. Evaluation metrics are essential for measuring the effectiveness of a model
 * and guiding the optimization process during training.
 *
 * Key functionalities include:
 *
 * 1. Obtaining the predicted class label from the model's output.
 * 2. Computing the F1-score, a measure that combines precision and recall, for
 *    a given output and true label.
 *
 * This header file serves as a central hub for working with evaluation metrics,
 * ensuring consistency and simplifying the integration of different metrics into
 * machine learning models.
 *
 * Usage examples:
 *
 * // Get the predicted class label from the model's output
 * float* output = ...; // Output from the model
 * int num_classes = 10; // Number of classes in the output
 * int predicted_label = get_prediction_accuracy(output, num_classes);
 *
 * // Compute the F1-score for a given output and true label
 * float* output = ...; // Output from the model
 * int true_label = 3; // True label for the input
 * int num_classes = 10; // Number of classes in the output
 * float f1_score = compute_f1_score(output, true_label, num_classes);
 */

#ifndef METRIC_H
#define METRIC_H

int get_prediction_accuracy(float* output, int num_classes);
float compute_f1_score(float* output, int label, int num_classes);

#endif // /* METRIC_H */