/* input/data.h 
 *
 * This file provides functions to load input data from images and resize it
 * to a specified size. It also provides a union structure to store the input
 * data in a format that can be processed by the neural network.
 * 
 */

#ifndef DATA_H
#define DATA_H

#include <stdint.h>
#include <stdbool.h>

#include "../utils/utils.h"

// Define the Input and DataType structures

// Define the data types supported by the library
typedef enum {
    Int,
    FLOAT32
} DataType;

typedef union {
        int ***int_data;
        float ***float32_data;
} InputData;

typedef enum {
    JPEG,
    JPG,
    PNG,
    GIF,
    TGA,
    BMP,
    PSD,
    HDR,
    PIC
    // Add support for other formats here
} ImageFormat;

// Load an image from file and store it in an InputData struct
InputData* load_input_data_from_image(const char *filename, const Dimensions *input_dimensions, DataType data_type);

// Load an image from disk with a specified format
InputData *load_image_data_with_format(const char *filename, const Dimensions *input_dimensions, DataType data_type, ImageFormat format);

// Load an JPEG image from disk as a float array
float*** loadFloatJPEG(const char* jpegFileName, int* width, int* height);

// Load an JPEG image from disk as a int array
int*** loadIntJPEG(const char* jpegFileName, int* width, int* height);

// Load a PNG image from disk as a float array
float*** loadFloatPNG(const char* pngFileName, int* width, int* height);

// Load a PNG image from disk as a int array
int*** loadIntPNG(const char* pngFileName, int* width, int* height);

// Load an image from disk as a float array
float*** loadFloatImage(const char* fileName, int* width, int* height, int* channels);

// Load an image from disk as a int array
int*** loadIntImage(const char* fileName, int* width, int* height, int* channels);

// Create an empty image
InputData* create_empty_input_data(int width, int height, int channels, DataType data_type, int fill_value);

// Resize an image
void resize_image(InputData **image_data_ptr, const Dimensions original_dimensions, Dimensions new_dimensions, DataType data_type);

// Preprocess the input data (unfinished)
// void preprocess_input(InputData* input_data, Dimensions input_shape);

// Free the memory associated with an image
void free_image_data(InputData *image_data, Dimensions dimensions, DataType data_type);

#endif /* Data_H */
