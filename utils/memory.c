/* utils/memory.c */

#include <stdlib.h>

#include "memory.h"

int**** malloc_4d_int_array(int dim1, int dim2, int dim3, int dim4) {
    int**** arr = (int****)malloc(dim1 * sizeof(int***));
    if (arr == NULL) {
        return NULL;
    }

    for (int i = 0; i < dim1; i++) {
        arr[i] = (int***)malloc(dim2 * sizeof(int**));
        if (arr[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(arr[j]);
            }
            free(arr);
            return NULL;
        }

        for (int j = 0; j < dim2; j++) {
            arr[i][j] = (int**)malloc(dim3 * sizeof(int*));
            if (arr[i][j] == NULL) {
                for (int k = 0; k < j; k++) {
                    free(arr[i][k]);
                }
                free(arr[i]);
                for (int l = 0; l < i; l++) {
                    free(arr[l]);
                }
                free(arr);
                return NULL;
            }

            for (int k = 0; k < dim3; k++) {
                arr[i][j][k] = (int*)malloc(dim4 * sizeof(int));
                if (arr[i][j][k] == NULL) {
                    for (int m = 0; m < k; m++) {
                        free(arr[i][j][m]);
                    }
                    free(arr[i][j]);
                    for (int l = 0; l < j; l++) {
                        free(arr[i][l]);
                    }
                    free(arr[i]);
                    for (int n = 0; n < i; n++) {
                        free(arr[n]);
                    }
                    free(arr);
                    return NULL;
                }
            }
        }
    }

    return arr;
}

float**** malloc_4d_float_array(int dim1, int dim2, int dim3, int dim4) {
    float**** arr = (float****)malloc(dim1 * sizeof(float***));
    if (arr == NULL) {
        return NULL;
    }

    for (int i = 0; i < dim1; i++) {
        arr[i] = (float***)malloc(dim2 * sizeof(float**));
        if (arr[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(arr[j]);
            }
            free(arr);
            return NULL;
        }

        for (int j = 0; j < dim2; j++) {
            arr[i][j] = (float**)malloc(dim3 * sizeof(float*));
            if (arr[i][j] == NULL) {
                for (int k = 0; k < j; k++) {
                    free(arr[i][k]);
                }
                free(arr[i]);
                for (int l = 0; l < i; l++) {
                    free(arr[l]);
                }
                free(arr);
                return NULL;
            }

            for (int k = 0; k < dim3; k++) {
                arr[i][j][k] = (float*)malloc(dim4 * sizeof(float));
                if (arr[i][j][k] == NULL) {
                    for (int m = 0; m < k; m++) {
                        free(arr[i][j][m]);
                    }
                    free(arr[i][j]);
                    for (int l = 0; l < j; l++) {
                        free(arr[i][l]);
                    }
                    free(arr[i]);
                    for (int n = 0; n < i; n++) {
                        free(arr[n]);
                    }
                    free(arr);
                    return NULL;
                }
            }
        }
    }

    return arr;
}

void free_4d_int_array(int**** arr, int dim1, int dim2, int dim3) {
    if (arr == NULL) {
        return;
    }

    for (int i = 0; i < dim1; i++) {
        if (arr[i] != NULL) {
            for (int j = 0; j < dim2; j++) {
                if (arr[i][j] != NULL) {
                    for (int k = 0; k < dim3; k++) {
                        if (arr[i][j][k] != NULL) {
                            free(arr[i][j][k]);
                        }
                    }
                    free(arr[i][j]);
                }
            }
            free(arr[i]);
        }
    }
    free(arr);
}

void free_4d_float_array(float**** arr, int dim1, int dim2, int dim3) {
    if (arr == NULL) {
        return;
    }

    for (int i = 0; i < dim1; i++) {
        if (arr[i] != NULL) {
            for (int j = 0; j < dim2; j++) {
                if (arr[i][j] != NULL) {
                    for (int k = 0; k < dim3; k++) {
                        if (arr[i][j][k] != NULL) {
                            free(arr[i][j][k]);
                        }
                    }
                    free(arr[i][j]);
                }
            }
            free(arr[i]);
        }
    }
    free(arr);
}

int*** malloc_3d_int_array(int dim1, int dim2, int dim3) {
    int*** arr = (int***)malloc(dim1 * sizeof(int**));
    if (arr == NULL) {
        return NULL;
    }

    for (int i = 0; i < dim1; i++) {
        arr[i] = (int**)malloc(dim2 * sizeof(int*));
        if (arr[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(arr[j]);
            }
            free(arr);
            return NULL;
        }

        for (int j = 0; j < dim2; j++) {
            arr[i][j] = (int*)malloc(dim3 * sizeof(int));
            if (arr[i][j] == NULL) {
                for (int k = 0; k < j; k++) {
                    free(arr[i][k]);
                }
                free(arr[i]);
            }
        }
    }

    return arr;
}

float*** malloc_3d_float_array(int dim1, int dim2, int dim3) {
    float*** arr = (float***)malloc(dim1 * sizeof(float**));
    if (arr == NULL) {
        return NULL;
    }

    for (int i = 0; i < dim1; i++) {
        arr[i] = (float**)malloc(dim2 * sizeof(float*));
        if (arr[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(arr[j]);
            }
            free(arr);
            return NULL;
        }

        for (int j = 0; j < dim2; j++) {
            arr[i][j] = (float*)malloc(dim3 * sizeof(float));
            if (arr[i][j] == NULL) {
                for (int k = 0; k < j; k++) {
                    free(arr[i][k]);
                }
                free(arr[i]);
                for (int l = 0; l < i; l++) {
                    free(arr[l]);
                }
                free(arr);
                return NULL;
            }
        }
    }

    return arr;
}

void free_3d_int_array(int*** arr, int dim1, int dim2) {
    if (arr == NULL) {
        return;
    }

    for (int i = 0; i < dim1; i++) {
        if (arr[i] != NULL) {
            for (int j = 0; j < dim2; j++) {
                if (arr[i][j] != NULL) {
                    free(arr[i][j]);
                }
            }
            free(arr[i]);
        }
    }
    free(arr);
}

void free_3d_float_array(float*** arr, int dim1, int dim2) {
    if (arr == NULL) {
        return;
    }

    for (int i = 0; i < dim1; i++) {
        if (arr[i] != NULL) {
            for (int j = 0; j < dim2; j++) {
                if (arr[i][j] != NULL) {
                    free(arr[i][j]);
                }
            }
            free(arr[i]);
        }
    }
    free(arr);
}

int** malloc_2d_int_array(int dim1, int dim2) {
    int** arr = (int**)malloc(dim1 * sizeof(int*));
    if (arr == NULL) {
        return NULL;
    }

    for (int i = 0; i < dim1; i++) {
        arr[i] = (int*)malloc(dim2 * sizeof(int));
        if (arr[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(arr[j]);
            }
            free(arr);
            return NULL;
        }
    }

    return arr;
}

float** malloc_2d_float_array(int dim1, int dim2) {
    float** arr = (float**)malloc(dim1 * sizeof(float*));
    if (arr == NULL) {
        return NULL;
    }

    for (int i = 0; i < dim1; i++) {
        arr[i] = (float*)malloc(dim2 * sizeof(float));
        if (arr[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(arr[j]);
            }
            free(arr);
            return NULL;
        }
    }

    return arr;
}

void free_2d_int_array(int** arr, int dim1) {
    if (arr == NULL) {
        return;
    }

    for (int i = 0; i < dim1; i++) {
        if (arr[i] != NULL) {
            free(arr[i]);
        }
    }
    free(arr);
}

void free_2d_float_array(float** arr, int dim1) {
    if (arr == NULL) {
        return;
    }

    for (int i = 0; i < dim1; i++) {
        if (arr[i] != NULL) {
            free(arr[i]);
        }
    }
    free(arr);
}

int* malloc_1d_int_array(int dim) {
    int* arr = (int*)malloc(dim * sizeof(int));
    if (arr == NULL) {
        return NULL;
    }

    return arr;
}

float* malloc_1d_float_array(int dim) {
    float* arr = (float*)malloc(dim * sizeof(float));
    if (arr == NULL) {
        return NULL;
    }

    return arr;
}   

void free_1d_int_array(int* arr) {
    if (arr == NULL) {
        return;
    }

    free(arr);
}

void free_1d_float_array(float* arr) {
    if (arr == NULL) {
        return;
    }

    free(arr);
}
