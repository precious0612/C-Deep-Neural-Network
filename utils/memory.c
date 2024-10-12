//
//  memory.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/15/24.
//

#include "memory.h"

#include <stdlib.h>
#include <string.h>

int** calloc_2d_int_array(int dim1, int dim2) {
    int i;

    int** intArray = (int**)calloc(dim1, sizeof(int*));
    if (intArray == NULL) {
        // Handle memory allocation failure
        return NULL;
    }

    // Step 2: Allocate memory for the actual integers in a contiguous block
    intArray[0] = (int*)calloc(dim1 * dim2, sizeof(int));
    if (intArray[0] == NULL) {
        // Handle memory allocation failure
        free(intArray);
        intArray = NULL;
        return NULL;
    }

    // Set up the pointers for rows
    for (i = 0; i < dim1; ++i) {
        if (i > 0) {
            intArray[i] = intArray[0] + i * dim2;
        }
    }

    return intArray;
}

float** calloc_2d_float_array(int dim1, int dim2) {
    int i;

    float** floatArray = (float**)calloc(dim1, sizeof(float*));
    if (floatArray == NULL) {
        // Handle memory allocation failure
        return NULL;
    }

    // Step 2: Allocate memory for the actual floats in a contiguous block
    floatArray[0] = (float*)calloc(dim1 * dim2, sizeof(float));
    if (floatArray[0] == NULL) {
        // Handle memory allocation failure
        free(floatArray);
        floatArray = NULL;
        return NULL;
    }

    // Set up the pointers for rows
    for (i = 0; i < dim1; ++i) {
        if (i > 0) {
            floatArray[i] = floatArray[0] + i * dim2;
        }
    }

    return floatArray;
}

int** copy_2d_int_array(int** src, int dim1, int dim2) {
    if (src == NULL) {
        return NULL;
    }

    int** dst = calloc_2d_int_array(dim1, dim2);
    if (dst == NULL) {
        return NULL;
    }

    memcpy(src[0], dst[0], dim1 * dim2 * sizeof(int));

    return dst;
}

float** copy_2d_float_array(float** src, int dim1, int dim2) {
    if (src == NULL) {
        return NULL;
    }

    float** dst = calloc_2d_float_array(dim1, dim2);
    if (dst == NULL) {
        return NULL;
    }

    memcpy(dst[0], src[0], dim1 * dim2 * sizeof(float));

    return dst;
}

void free_2d_int_array(int** intArray) {
    if (intArray) {
        if (intArray[0]) {
            free(intArray[0]);
            intArray[0] = NULL;
        }
        free(intArray);
        intArray = NULL;
    }
}

void free_2d_float_array(float** floatArray) {
    if (floatArray) {
        if (floatArray[0]) {
            free(floatArray[0]);
            floatArray[0] = NULL;
        }
        free(floatArray);
        floatArray = NULL;
    }
}

int*** calloc_3d_int_array(int dim1, int dim2, int dim3) {
    int i, j;

    int*** intArray = (int***)calloc(dim1, sizeof(int**));
    if (intArray == NULL) {
        // Handle memory allocation failure
        return NULL;
    }

    // Step 2: Allocate memory for the pointers to the columns within each row
    intArray[0] = (int**)calloc(dim1 * dim2, sizeof(int*));
    if (intArray[0] == NULL) {
        // Handle memory allocation failure
        free(intArray);
        intArray = NULL;
        return NULL;
    }

    // Step 3: Allocate memory for the actual integers in a contiguous block
    intArray[0][0] = (int*)calloc(dim1 * dim2 * dim3, sizeof(int));
    if (intArray[0][0] == NULL) {
        // Handle memory allocation failure
        free(intArray[0]);
        intArray[0] = NULL;
        free(intArray);
        intArray = NULL;
        return NULL;
    }

    // Set up the pointers for rows and columns
    for (i = 0; i < dim1; ++i) {
        if (i > 0) {
            intArray[i] = intArray[0] + i * dim2;
        }

        for (j = 0; j < dim2; ++j) {
            if (i == 0 && j > 0) {
                intArray[0][j] = intArray[0][0] + j * dim3;
            } else if (i > 0) {
                intArray[i][j] = intArray[0][0] + (i * dim2 + j) * dim3;
            }
        }
    }

    return intArray;
}

float*** calloc_3d_float_array(int dim1, int dim2, int dim3) {
    int i, j;

    float*** floatArray = (float***)calloc(dim1, sizeof(float**));
    if (floatArray == NULL) {
        // Handle memory allocation failure
        return NULL;
    }

    // Step 2: Allocate memory for the pointers to the columns within each row
    floatArray[0] = (float**)calloc(dim1 * dim2, sizeof(float*));
    if (floatArray[0] == NULL) {
        // Handle memory allocation failure
        free(floatArray);
        floatArray = NULL;
        return NULL;
    }

    // Step 3: Allocate memory for the actual integers in a contiguous block
    floatArray[0][0] = (float*)calloc(dim1 * dim2 * dim3, sizeof(float));
    if (floatArray[0][0] == NULL) {
        // Handle memory allocation failure
        free(floatArray[0]);
        floatArray[0] = NULL;
        free(floatArray);
        floatArray = NULL;
        return NULL;
    }

    // Set up the pointers for rows and columns
    for (i = 0; i < dim1; ++i) {
        if (i > 0) {
            floatArray[i] = floatArray[0] + i * dim2;
        }

        for (j = 0; j < dim2; ++j) {
            if (i == 0 && j > 0) {
                floatArray[0][j] = floatArray[0][0] + j * dim3;
            } else if (i > 0) {
                floatArray[i][j] = floatArray[0][0] + (i * dim2 + j) * dim3;
            }
        }
    }

    return floatArray;
}

int*** copy_3d_int_array(int*** src, int dim1, int dim2, int dim3) {
    if (src == NULL) {
        return NULL;
    }

    int*** dst = calloc_3d_int_array(dim1, dim2, dim3);
    if (dst == NULL) {
        return NULL;
    }

    int* src_p = &src[0][0][0];
    int* dst_p = &dst[0][0][0];

    memcpy(dst_p, src_p, dim1 * dim2 * dim3 * sizeof(int));

    return dst;
}

float*** copy_3d_float_array(float*** src, int dim1, int dim2, int dim3) {
    if (src == NULL) {
        return NULL;
    }

    float*** dst = calloc_3d_float_array(dim1, dim2, dim3);
    if (dst == NULL) {
        return NULL;
    }

    float* src_p = &src[0][0][0];
    float* dst_p = &dst[0][0][0];

    memcpy(dst_p, src_p, dim1 * dim2 * dim3 * sizeof(float));

    return dst;
}

float*** copy_3d_float_array_from_int(int*** src, int dim1, int dim2, int dim3) {
    if (src == NULL) {
        return NULL;
    }

    float*** dst = calloc_3d_float_array(dim1, dim2, dim3);
    if (dst == NULL) {
        return NULL;
    }

    int*   src_p = &src[0][0][0];
    float* dst_p = &dst[0][0][0];

    for (int index = 0; index < dim1 * dim2 * dim3; ++index) {
        dst_p[index] = (float)src_p[index];
    }

    return dst;
}

void free_3d_int_array(int*** intArray) {
    if (intArray) {
            if (intArray[0]) {
                if (intArray[0][0]) {
                    free(intArray[0][0]);
                    intArray[0][0] = NULL;
                }
                free(intArray[0]);
                intArray[0] = NULL;
            }
            free(intArray);
            intArray = NULL;
        }
}

void free_3d_float_array(float*** floatArray) {
    if (floatArray) {
        if (floatArray[0]) {
            if (floatArray[0][0]) {
                free(floatArray[0][0]);
                floatArray[0][0] = NULL;
            }
            free(floatArray[0]);
            floatArray[0] = NULL;
        }
        free(floatArray);
        floatArray = NULL;
    }
}

int**** calloc_4d_int_array(int dim1, int dim2, int dim3, int dim4) {
    int i, j, k;

    int**** intArray = (int****)calloc(dim1, sizeof(int***));
    if (intArray == NULL) {
        // Handle memory allocation failure
        return NULL;
    }

    intArray[0] = (int***)calloc(dim1 * dim2, sizeof(int**));
    if (intArray[0] == NULL) {
        free(intArray);
        return NULL;
    }

    intArray[0][0] = (int**)calloc(dim1 * dim2 * dim3, sizeof(int*));
    if (intArray[0][0] == NULL) {
        free(intArray[0]);
        free(intArray);
        return NULL;
    }

    intArray[0][0][0] = (int*)calloc(dim1 * dim2 * dim3 * dim4, sizeof(int));
    if (intArray[0][0][0] == NULL) {
        free(intArray[0][0]);
        free(intArray[0]);
        free(intArray);
        return NULL;
    }

    for (i = 0; i < dim1; ++i) {
        if (i > 0) {
            intArray[i] = intArray[0] + i * dim2;
        }

        for (j = 0; j < dim2; ++j) {
            if (i == 0 && j > 0) {
                intArray[0][j] = intArray[0][0] + j * dim3;
            } else if (i > 0) {
                intArray[i][j] = intArray[0][0] + (i * dim2 + j) * dim3;
            }

            for (k = 0; k < dim3; ++k) {
                if (i == 0 && j == 0 && k > 0) {
                    intArray[0][0][k] = intArray[0][0][0] + k * dim4;
                } else if (i > 0 || j > 0) {
                    intArray[i][j][k] = intArray[0][0][0] + ((i * dim2 + j) * dim3 + k) * dim4;
                }
            }
        }
    }

    return intArray;
}

float**** calloc_4d_float_array(int dim1, int dim2, int dim3, int dim4) {
    int i, j, k;

    float**** floatArray = (float****)calloc(dim1, sizeof(float***));
    if (floatArray == NULL) {
        // Handle memory allocation failure
        return NULL;
    }

    floatArray[0] = (float***)calloc(dim1 * dim2, sizeof(float**));
    if (floatArray[0] == NULL) {
        free(floatArray);
        return NULL;
    }

    floatArray[0][0] = (float**)calloc(dim1 * dim2 * dim3, sizeof(float*));
    if (floatArray[0][0] == NULL) {
        free(floatArray[0]);
        free(floatArray);
        return NULL;
    }

    floatArray[0][0][0] = (float*)calloc(dim1 * dim2 * dim3 * dim4, sizeof(float));
    if (floatArray[0][0][0] == NULL) {
        free(floatArray[0][0]);
        free(floatArray[0]);
        free(floatArray);
        return NULL;
    }

    for (i = 0; i < dim1; ++i) {
        if (i > 0) {
            floatArray[i] = floatArray[0] + i * dim2;
        }

        for (j = 0; j < dim2; ++j) {
            if (i == 0 && j > 0) {
                floatArray[0][j] = floatArray[0][0] + j * dim3;
            } else if (i > 0) {
                floatArray[i][j] = floatArray[0][0] + (i * dim2 + j) * dim3;
            }

            for (k = 0; k < dim3; ++k) {
                if (i == 0 && j == 0 && k > 0) {
                    floatArray[0][0][k] = floatArray[0][0][0] + k * dim4;
                } else if (i > 0 || j > 0) {
                    floatArray[i][j][k] = floatArray[0][0][0] + ((i * dim2 + j) * dim3 + k) * dim4;
                }
            }
        }
    }

    return floatArray;
}

int**** copy_4d_int_array(int**** src, int dim1, int dim2, int dim3, int dim4) {
    if (src == NULL) {
        return NULL;
    }

    int**** dst = calloc_4d_int_array(dim1, dim2, dim3, dim4);
    if (dst == NULL) {
        return NULL;
    }

    int* src_p = &src[0][0][0][0];
    int* dst_p = &dst[0][0][0][0];

    memcpy(dst_p, src_p, dim1 * dim2 * dim3 * dim4 * sizeof(int));

    return dst;
}

float**** copy_4d_float_array(float**** src, int dim1, int dim2, int dim3, int dim4) {
    if (src == NULL) {
        return NULL;
    }

    float**** dst = calloc_4d_float_array(dim1, dim2, dim3, dim4);
    if (dst == NULL) {
        return NULL;
    }

    float* src_p = &src[0][0][0][0];
    float* dst_p = &dst[0][0][0][0];

    memcpy(dst_p, src_p, dim1 * dim2 * dim3 * dim4 * sizeof(float));

    return dst;
}

void free_4d_int_array(int**** intArray) {
    if (intArray) {
        if (intArray[0]) {
            if (intArray[0][0]) {
                if (intArray[0][0][0]) {
                    free(intArray[0][0][0]);
                    intArray[0][0][0] = NULL;
                }
                free(intArray[0][0]);
                intArray[0][0] = NULL;
            }
            free(intArray[0]);
            intArray[0] = NULL;
        }
        free(intArray);
        intArray = NULL;
    }
}

void free_4d_float_array(float**** floatArray) {
    if (floatArray) {
        if (floatArray[0]) {
            if (floatArray[0][0]) {
                if (floatArray[0][0][0]) {
                    free(floatArray[0][0][0]);
                    floatArray[0][0][0] = NULL;
                }
                free(floatArray[0][0]);
                floatArray[0][0] = NULL;
            }
            free(floatArray[0]);
            floatArray[0] = NULL;
        }
        free(floatArray);
        floatArray = NULL;
    }
}

float* concatenate_float_array(float* a, int size_a, float* b, int size_b) {
    float *result = malloc((size_a + size_b) * sizeof(float));
    if (result == NULL) {
        return NULL;
    }

    memcpy(result, a, size_a * sizeof(float));
    memcpy(result + size_a, b, size_b * sizeof(float));

    return result;
}
