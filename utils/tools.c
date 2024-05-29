//
//  tools.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/18/24.
//

#include "tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_progress_bar(float progress, int progress_bar_length) {
    // Validate input
    if (progress < 0.0f || progress > 1.0f) {
        fprintf(stderr, "Error: Invalid progress value (should be between 0.0 and 1.0)\n");
        return;
    }

    if (progress_bar_length <= 0 || progress_bar_length > MAX_PROGRESS_BAR_LENGTH) {
        fprintf(stderr, "Error: Invalid progress bar length (should be between 1 and %d)\n", MAX_PROGRESS_BAR_LENGTH);
        return;
    }

    int filled_length = (int)(progress * progress_bar_length);
    int remaining_length = progress_bar_length - filled_length;

    printf("[");
    for (int i = 0; i < filled_length; i++) {
        printf("#");
    }
    for (int i = 0; i < remaining_length; i++) {
        printf("-");
    }
    printf("] %3.0f%%\n", progress * 100);
    fflush(stdout);

    if (progress >= 1.0) {
        printf("\n");
    }
}

float rand_uniform(float min, float max) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    return min + (max - min) * (float)rand() / RAND_MAX;
}

int rand_int(int min, int max) {
    return rand_uniform(min, max + 1);
}
