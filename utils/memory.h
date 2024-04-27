/* utils/memory.h */

#ifndef MEMORY_H
#define MEMORY_H

int**** malloc_4d_int_array(int dim1, int dim2, int dim3, int dim4);
float**** malloc_4d_float_array(int dim1, int dim2, int dim3, int dim4);
void free_4d_int_array(int**** arr, int dim1, int dim2, int dim3);
void free_4d_float_array(float**** arr, int dim1, int dim2, int dim3);
int*** malloc_3d_int_array(int dim1, int dim2, int dim3);
float*** malloc_3d_float_array(int dim1, int dim2, int dim3);
void free_3d_int_array(int*** arr, int dim1, int dim2);
void free_3d_float_array(float*** arr, int dim1, int dim2);
int** malloc_2d_int_array(int dim1, int dim2);
float** malloc_2d_float_array(int dim1, int dim2);
void free_2d_int_array(int** arr, int dim1);
void free_2d_float_array(float** arr, int dim1);
int* malloc_1d_int_array(int dim1);
float* malloc_1d_float_array(int dim1);
void free_1d_int_array(int* arr);
void free_1d_float_array(float* arr);

#endif /* MEMORY_H */
