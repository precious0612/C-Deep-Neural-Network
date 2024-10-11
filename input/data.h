//
//  data.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/15/24.
//

#ifndef data_h
#define data_h

#include "../utils/utils.h"

// MARK: - Define the Input and DataType structures

// TODO: Define the data types supported by the library
typedef enum {
    SINT32,    // most OS int will be 32 bytes (prevent confliction with windows.h)
    FLOAT32
}DataType;

// TODO: Define InputData structure
typedef  union {
    int   ***int_data;       // shape: {width, height, channels}
    float ***float32_data;   // shape: {width, height, channels}
}InputData;

// TODO: Define the imges types for processing
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
}ImageFormat;

// MARK: - Method Declarations

// TODO: User-level Interface

/// Loads image data from a file into an `InputData` structure, automatically determining the image format based on the file extension.
///
/// - Parameters:
///   - filename: Path to the image file.
///   - input_dimensions: A pointer to a `Dimensions` structure specifying the desired dimensions of the input data.
///   - data_type: The type of data that the image will be converted to (e.g., `SINT32` or `FLOAT32`).
/// - Returns: a pointer to the `InputData` structure containing the loaded image data, or `NULL` if the image could not be loaded.
///
/// - Logic for Determining Image Format:
///   - The function identifies the image format by examining the file extension. If no extension is found, it defaults to JPEG.
///   - Supported formats include JPEG, PNG, GIF, TGA, BMP, PSD, HDR, and PIC.
///   - If the file extension does not match any known formats, the function defaults to PIC.
///
/// - Example Usage:
///     ```c
///     Dimensions img_dims = {640, 480, 3};
///     InputData* image_data = load_input_data_from_image("path/to/image.jpg", &img_dims, SINT32);
///     if (image_data == NULL) {
///         // Handle error
///     }
///     ```
///
InputData* load_input_data_from_image(const char *filename, const Dimensions *input_dimensions, DataType data_type);

/// Creates an empty `InputData` structure with specified dimensions and data type, initializing pixel values to a given fill value.
///
/// - Parameters:
///   - width: The width of the image data.
///   - height: The height of the image data.
///   - channels: The number of channels in the imge data.
///   - data_type: The type of data stored in the image data (e.g., `SINT32` or `FLOAT32`).
///   - fill_value: The value to initialize each pixel with.
/// - Throws: If memory allocation for the `InputData` structure fails, an error message is printed to `stderr`, and `NULL` is returned. If memory allocation for the `int_data` or `float32_data` fails at any point, an error message is printed to `stderr`, all previously allocated memory is freed, and `NULL` is returned.
/// - Returns: a pointer to the newly created `InputData` structure, or `NULL` if memory allocation fails.
///
/// - Example Usage:
///     ```c
///     InputData* image_data = create_empty_input_data(640, 480, 3, SINT32, 0);
///     if (image_data == NULL) {
///         // Handle error
///     }
///     ```
///
InputData* create_empty_input_data(int width, int height, int channels, DataType data_type, int fill_value);

/// Resizes an image to new dimensions using bilinear interpolation.
/// - Parameters:
///   - image_data_ptr: A pointer to a pointer of the `InputData` structure containing the original image data.
///   - original_dimensions: The original dimensions of the image.
///   - new_dimensions: The new dimensions to which the image should be resized.
///   - data_type: The type of data stored in the image data (e.g., `SINT32` or `FLOAT32`).
/// - Returns: `0` on successful resize, `-1` if the provided dimensions are invalid (non-positive values) and `-2` if memory allocation for the resized image fails.
///
/// - Resizing Algorithm
///   - The function calculates scaling factors based on the ratio of new dimensions to original dimensions.
///   - It performs bilinear interpolation to calculate the pixel values at the new positions.
///   - The indices for the interpolation are clamped to the bounds of the original image to avoid accessing out-of-range memory.
///
/// - Example Usage:
///     ```c
///     Dimensions orig_dims = {640, 480, 3};
///     Dimensions new_dims = {800, 600, 3};
///     int result = resize_image(&image_data, orig_dims, new_dims, SINT32);
///     if (result != 0) {
///         // Handle error
///     }
///     ```
///
int resize_image(InputData **image_data_ptr, const Dimensions original_dimensions, Dimensions new_dimensions, DataType data_type);


/// Copies the image data to another `InputData`
/// - Parameters:
///   - src: A pointer to the source `InputData` structure.
///   - dimensions: The dimensions of the source image.
///   - data_type: The type of data stored in the image data (e.g., `SINT32` or `FLOAT32`).
/// - Returns: A pointer to the new `InputData` structure stored the source image data.
///
/// - Example Usage:
///     ```c
///     InputData* dst_data = copy_image_data(src_data, src_dims, src_data_type);
///     free_image_data(src_data, src_dims, src_data_type);
///     ```
///
InputData* copy_image_data(InputData* src, Dimensions dimensions, DataType data_type);

/// Frees the allocated memory for an image data structure.
/// - Parameters:
///   - image_data: A pointer to the `InputData` structure containing the image data to be freed.
///   - dimensions: The dimensions of the image, used to determine the number of rows in the data arrays.
///   - data_type: The type of data stored in the image data (e.g., `SINT32` or `FLOAT32`).
///
/// - Memory Deallocation Algorithm:
///   - The function first checks if the `image_data` pointer is `NULL`, and if so, returns immediately.
///   - It then proceeds to free the memory allocated for the pixel values, depending on the `data_type`.
///   - For each data type, it frees the contiguous block of memory for pixel values and the row pointers.
///   - Finally, it frees the `InputData` structure itself and sets the pointer to `NULL`.
///
/// - Example Usage:
///     ```c
///     InputData* image_data = create_image_data(orig_dims, SINT32);
///     // ... image manipulation code ...
///     free_image_data(image_data, orig_dims, SINT32);
///     ```
///
void free_image_data(InputData *image_data, Dimensions dimensions, DataType data_type);

// TODO: Low-level Interface

/// Load an image from disk with a specified format
InputData* load_image_data_with_format(const char* filename, const Dimensions *input_dimensions, DataType data_type, ImageFormat format);

#endif /* data_h */
