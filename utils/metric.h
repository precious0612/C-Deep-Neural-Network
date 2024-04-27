/* utils/metric.h */

#ifndef METRIC_H
#define METRIC_H

int get_prediction_accuracy(float* output, int num_classes);
float compute_f1_score(float* output, int label, int num_classes);

#endif // /* METRIC_H */