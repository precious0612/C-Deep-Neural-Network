/* utils/loss.c */

void categorical_crossentropy_loss(float*** output, int label, int batch_size, int height, int num_classes, float*** output_grad) {
    for (int j = 0; j < height; j++) {
        for (int k = 0; k < batch_size; k++) {
            for (int c = 0; c < num_classes; c++) {
                if (c == label) {
                    output_grad[j][k][c] = output[j][k][c] - 1.0f;
                } else {
                    output_grad[j][k][c] = output[j][k][c];
                }
            }
        }
    }
}

void mean_squared_error_loss(float*** output, int label, int batch_size, int height, int num_classes, float*** output_grad) {
    for (int j = 0; j < height; j++) {
        for (int k = 0; k < batch_size; k++) {
            for (int c = 0; c < num_classes; c++) {
                if (c == label) {
                    output_grad[j][k][c] = output[j][k][c] - 1.0f;
                } else {
                    output_grad[j][k][c] = output[j][k][c];
                }
            }
        }
    }
}