/* utils/example.c */

#include <stdio.h>

#include "memory.h"

int main() {
    int dim1 = 2, dim2 = 3, dim3 = 4;
    float*** arr = malloc_3d_float_array(dim1, dim2, dim3);
    if (arr == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // Use the 3D array here
    // ...

    free_3d_float_array(arr, dim1, dim2);
    return 0;
}