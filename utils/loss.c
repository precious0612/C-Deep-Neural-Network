/* utils/loss.c */

#include <math.h>

float categorical_crossentropy_loss(float* output, int label, int num_classes) {
    float loss = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        if (c == label) {
            loss += -log(output[c]);
        } else {
            loss += -log(1.0f - output[c]);
        }
    }

    return loss / num_classes;
}

float mean_squared_error_loss(float* output, int label, int num_classes) {
    float loss = 0.0f;
        float diff = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            if (c == label) {
                diff = 1.0f - output[c];
            } else {
                diff = -output[c];
            }
            loss += diff * diff;
        }
    return loss;
}
