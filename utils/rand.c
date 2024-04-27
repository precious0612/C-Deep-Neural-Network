/* utils/rand.c */

#include <stdlib.h>
#include <time.h>

#include "rand.h"

// #define RAND_MAX 32767

float rand_uniform(float min, float max) {
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    return min + (max - min) * (float)rand() / RAND_MAX;
}

int rand_int(int min, int max) {
    return rand_uniform(min, max + 1);
}
