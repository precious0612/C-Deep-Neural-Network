/* utils/metric.h */

#ifndef METRIC_H
#define METRIC_H

int get_prediction_accuracy(float*** output, int num_classes, Dimensions output_dim);
float compute_f1_score(float*** output, int label, int num_classes, Dimensions output_dim);

#endif // /* METRIC_H */