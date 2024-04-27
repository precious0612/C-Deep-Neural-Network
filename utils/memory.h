/* utils/memory.h
 *
 * This file provides a set of utility functions for dynamically allocating and
 * deallocating multi-dimensional arrays for both integer and floating-point data
 * types. These functions are essential for working with tensors and matrices
 * commonly used in deep learning applications.
 *
 * Key functionalities include:
 *
 * 1. Allocating and freeing 4D arrays for integers and floats.
 * 2. Allocating and freeing 3D arrays for integers and floats.
 * 3. Allocating and freeing 2D arrays for integers and floats.
 * 4. Allocating and freeing 1D arrays for integers and floats.
 *
 * These functions ensure proper memory management and help prevent common issues
 * such as memory leaks and buffer overflows. By providing a consistent and
 * easy-to-use interface for memory allocation and deallocation, this header file
 * simplifies the handling of multi-dimensional arrays throughout the codebase.
 *
 * Usage examples:
 *
 * // Allocating a 4D float array
 * float**** tensor = malloc_4d_float_array(2, 3, 4, 5);
 * // Use the tensor
 * ...
 * // Free the tensor
 * free_4d_float_array(tensor, 2, 3, 4);
 *
 * // Allocating a 3D int array
 * int*** matrix = malloc_3d_int_array(10, 20, 30);
 * // Use the matrix
 * ...
 * // Free the matrix
 * free_3d_int_array(matrix, 10, 20);
 */

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
