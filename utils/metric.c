/* utils/metric.c */

#include <math.h>
// #define INFINITY 1e8

int get_prediction_accuracy(float* output, int num_classes) {
    float max_value = -INFINITY;
    int prediction = -1;

    for (int c = 0; c < num_classes; c++) {
        float value = output[c];
        if (value > max_value) {
            max_value = value;
            prediction = c;
        }
    }

    return prediction;
}

float compute_f1_score(float* output, int label, int num_classes) {
    float true_positive = 0.0f;
    float false_positive = 0.0f;
    float false_negative = 0.0f;

    for (int k = 0; k < num_classes; k++) {
        float predicted_value = output[label];
        float true_value = (k == label) ? 1.0f : 0.0f;

        if (predicted_value > 0.5f) {
            if (true_value == 1.0f) {
                true_positive++;
            } else {
                false_positive++;
            }
        } else {
            if (true_value == 1.0f) {
                false_negative++;
            }
        }
    }

    float precision = true_positive / (true_positive + false_positive);
    float recall = true_positive / (true_positive + false_negative);
    float f1_score = 2 * precision * recall / (precision + recall);

    return f1_score;
}
