//
//  memory.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/15/24.
//

#ifndef memory_h
#define memory_h

// MARK: 2D memory tools

int** calloc_2d_int_array(int dim1, int dim2);
float** calloc_2d_float_array(int dim1, int dim2);

int** copy_2d_int_array(int** src, int dim1, int dim2);
float** copy_2d_float_array(float** src, int dim1, int dim2);

void free_2d_int_array(int** intArray);
void free_2d_float_array(float** floatArray);

// MARK: 3D memory tools

int*** calloc_3d_int_array(int dim1, int dim2, int dim3);
float*** calloc_3d_float_array(int dim1, int dim2, int dim3);

int*** copy_3d_int_array(int*** src, int dim1, int dim2, int dim3);
float*** copy_3d_float_array(float*** src, int dim1, int dim2, int dim3);
float*** copy_3d_float_array_from_int(int*** src, int dim1, int dim2, int dim3);

void free_3d_int_array(int*** intArray);
void free_3d_float_array(float*** floatArray);

// MARK: 4D memory tools

int**** calloc_4d_int_array(int dim1, int dim2, int dim3, int dim4);
float**** calloc_4d_float_array(int dim1, int dim2, int dim3, int dim4);

int**** copy_4d_int_array(int**** src, int dim1, int dim2, int dim3, int dim4);
float**** copy_4d_float_array(float**** src, int dim1, int dim2, int dim3, int dim4);

void free_4d_int_array(int**** intArray);
void free_4d_float_array(float**** floatArray);

// MARK: array tools

float* concatenate_float_array(float* a, int size_a, float* b, int size_b);

#endif /* memory_h */
