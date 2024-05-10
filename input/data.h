/* input/data.h
 *
 * This file provides functions to load input data from images and resize it
 * to a specified size. It also provides a union structure to store the input
 * data in a format that can be processed by the neural network.
 *
 * The InputData union is used to store the input data in either integer or
 * float format, depending on the specified DataType. The available data types
 * are INT (integer) and FLOAT32 (32-bit floating-point).
 *
 * The supported image formats include JPEG, PNG, GIF, TGA, BMP, PSD, HDR, and
 * a fallback format 'PIC' for other formats.
 *
 * This file provides the following functionality:
 *
 * 1. Loading input data from various image formats (JPEG, PNG, etc.)
 * 2. Resizing the input data to a specified size using interpolation
 * 3. Creating empty input data structures with optional initialization
 * 4. Freeing the memory allocated for input data
 */

#ifndef DATA_H
#define DATA_H

#include <stdint.h>
#include <stdbool.h>

#include "../utils/utils.h"

// Define the Input and DataType structures

// Define the data types supported by the library
typedef enum {
    INT,
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

/*
 * Usage example:
 *
 * Dimensions input_dimensions = {28, 28, 1};
 * InputData* image_data = load_input_data_from_image("image.jpg", &input_dimensions, FLOAT32);
 * if (image_data == NULL) {
 *     // Handle error
 * }
 * // Process image_data
 * free_image_data(image_data, input_dimensions, FLOAT32);
 */
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

/*
 * Usage example:
 *
 * InputData* image_data = create_empty_input_data(28, 28, 1, FLOAT32, 0.0f);
 * if (image_data == NULL) {
 *     // Handle error
 * }
 * // Process image_data
 * free_image_data(image_data, (Dimensions){28, 28, 1}, FLOAT32);
 */
InputData* create_empty_input_data(int width, int height, int channels, DataType data_type, int fill_value);

/*
 * Usage example:
 *
 * Dimensions original_dimensions = {32, 32, 3};
 * Dimensions new_dimensions = {28, 28, 3};
 * resize_image(&image_data, original_dimensions, new_dimensions, FLOAT32);
 */
void resize_image(InputData **image_data_ptr, const Dimensions original_dimensions, Dimensions new_dimensions, DataType data_type);

// Preprocess the input data (unfinished)
// void preprocess_input(InputData* input_data, Dimensions input_shape);

/*
 * Usage example:
 *
 * Dimensions dimensions = {28, 28, 1};
 * free_image_data(image_data, dimensions, FLOAT32);
 */
void free_image_data(InputData *image_data, Dimensions dimensions, DataType data_type);

#endif /* Data_H */
