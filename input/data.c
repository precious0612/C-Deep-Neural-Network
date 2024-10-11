//
//  data.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/15/24.
//

#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <turbojpeg.h>
#include <png.h>

// TODO: stb_image for other formats
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE__RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

// MARK: - Implementations

// TODO: Load Singal Image

InputData* load_input_data_from_image(const char* filename, const Dimensions* input_dimensions, DataType data_type) {
    // Determine image format based on filename extension
    const char* ext = strrchr(filename, '.');
    ImageFormat format;
    if (ext == NULL) {
        // Unable to determine format, default to JPEG
        format = JPEG;
    } else {
        ext++; // Skip the dot
        if (strcmp(ext, "jpg") == 0 || strcmp(ext, "jpeg") == 0) {
            format = JPG;
        } else if (strcmp(ext, "png") == 0) {
            format = PNG;
        } else if (strcmp(ext, "gif") == 0) {
            format = GIF;
        } else if (strcmp(ext, "tga") == 0) {
            format = TGA;
        } else if (strcmp(ext, "bmp") == 0) {
            format = BMP;
        } else if (strcmp(ext, "psd") == 0) {
            format = PSD;
        } else if (strcmp(ext, "hdr") == 0) {
            format = HDR;
        } else if (strcmp(ext, "pic") == 0) {
            format = PIC;
        } else {
            // Default to PIC for unknown formats
            format = PIC;
        }
    }

    // Delegate to original load_image_data function with the determined format and provided input dimensions
    return load_image_data_with_format(filename, input_dimensions, data_type, format);
}

// TODO: The methods loading image to data

static float*** loadFloatJPEG(const char* jpegFileName, int* width, int* height) {
    tjhandle jpegDecompressor = tjInitDecompress();
    if (!jpegDecompressor) {
        fprintf(stderr, "Error initializing JPEG decompressor\n");
        return NULL;
    }

    FILE* jpegFile = fopen(jpegFileName, "rb");
    if (!jpegFile) {
        fprintf(stderr, "Error opening JPEG file\n");
        tjDestroy(jpegDecompressor);
        return NULL;
    }

    fseek(jpegFile, 0, SEEK_END);
    long jpegFileSize = ftell(jpegFile);
    rewind(jpegFile);

    unsigned char* jpegBuffer = (unsigned char*)malloc(jpegFileSize);
    if (!jpegBuffer) {
        fprintf(stderr, "Error reading JPEG file!\n");
        free(jpegFile);
        jpegFile = NULL;
        tjDestroy(jpegDecompressor);
        return NULL;
    }

    if (fread(jpegBuffer, 1, jpegFileSize, jpegFile) != jpegFileSize) {
        fprintf(stderr, "Error reading JPEG file\n");
        free(jpegBuffer);
        jpegBuffer = NULL;
        fclose(jpegFile);
        tjDestroy(jpegDecompressor);
        return NULL;
    }
    fclose(jpegFile);

    int jpegSubsamp;
    int decompressStatus = tjDecompressHeader2(jpegDecompressor, jpegBuffer, jpegFileSize, width, height, &jpegSubsamp);
    if (decompressStatus != 0) {
        fprintf(stderr, "Error decompressing JPEG header\n");
        free(jpegBuffer);
        jpegBuffer = NULL;
        tjDestroy(jpegDecompressor);
        return NULL;
    }

    int pixelFormat = TJPF_RGB;
    unsigned char* rgbBuffer = (unsigned char*)malloc((*width) * (*height) * tjPixelSize[pixelFormat]);
    if (!rgbBuffer) {
        fprintf(stderr, "Error allocating memory for RGB buffer\n");
        free(jpegBuffer);
        jpegBuffer = NULL;
        tjDestroy(jpegDecompressor);
        return NULL;
    }

    decompressStatus = tjDecompress2(jpegDecompressor, jpegBuffer, jpegFileSize, rgbBuffer, *width, 0, *height, pixelFormat, 0);
    if (decompressStatus != 0) {
        fprintf(stderr, "Error decompressing JPEG image\n");
        free(jpegBuffer);
        free(rgbBuffer);
        jpegBuffer = NULL;
        rgbBuffer = NULL;
        tjDestroy(jpegDecompressor);
        return NULL;
    }

    free(jpegBuffer);
    jpegBuffer = NULL;
    tjDestroy(jpegDecompressor);

    float*** floatArray = calloc_3d_float_array(*width, *height, 3);
    if (!floatArray) {
        fprintf(stderr, "Error allocating memory for float array\n");
        free(rgbBuffer);
        rgbBuffer = NULL;
        return NULL;
    }

    float *p = &floatArray[0][0][0];
    for (int index = 0; index < *height * *width * 3; ++index) {
        p[index] = (float)rgbBuffer[index];
    }

    free(rgbBuffer);
    rgbBuffer = NULL;

    return floatArray;
}

static int*** loadIntJPEG(const char* jpegFileName, int* width, int* height) {
    tjhandle jpegDecompressor = tjInitDecompress();
    if (!jpegDecompressor) {
        fprintf(stderr, "Error initializing JPEG decompressor\n");
        return NULL;
    }

    FILE* jpegFile = fopen(jpegFileName, "rb");
    if (!jpegFile) {
        fprintf(stderr, "Error opening JPEG file\n");
        tjDestroy(jpegDecompressor);
        return NULL;
    }

    fseek(jpegFile, 0, SEEK_END);
    long jpegFileSize = ftell(jpegFile);
    rewind(jpegFile);

    unsigned char* jpegBuffer = (unsigned char*)malloc(jpegFileSize);
    if (!jpegBuffer) {
        fprintf(stderr, "Error reading JPEG file!\n");
        free(jpegFile);
        jpegFile = NULL;
        tjDestroy(jpegDecompressor);
        return NULL;
    }

    if (fread(jpegBuffer, 1, jpegFileSize, jpegFile) != jpegFileSize) {
        fprintf(stderr, "Error reading JPEG file\n");
        free(jpegBuffer);
        jpegBuffer = NULL;
        fclose(jpegFile);
        tjDestroy(jpegDecompressor);
        return NULL;
    }
    fclose(jpegFile);

    int jpegSubsamp;
    int decompressStatus = tjDecompressHeader2(jpegDecompressor, jpegBuffer, jpegFileSize, width, height, &jpegSubsamp);
    if (decompressStatus != 0) {
        fprintf(stderr, "Error decompressing JPEG header\n");
        free(jpegBuffer);
        jpegBuffer = NULL;
        tjDestroy(jpegDecompressor);
        return NULL;
    }

    int pixelFormat = TJPF_RGB;
    unsigned char* rgbBuffer = (unsigned char*)malloc((*width) * (*height) * tjPixelSize[pixelFormat]);
    if (!rgbBuffer) {
        fprintf(stderr, "Error allocating memory for RGB buffer\n");
        free(jpegBuffer);
        jpegBuffer = NULL;
        tjDestroy(jpegDecompressor);
        return NULL;
    }

    decompressStatus = tjDecompress2(jpegDecompressor, jpegBuffer, jpegFileSize, rgbBuffer, *width, 0, *height, pixelFormat, 0);
    if (decompressStatus != 0) {
        fprintf(stderr, "Error decompressing JPEG image\n");
        free(jpegBuffer);
        free(rgbBuffer);
        jpegBuffer = NULL;
        rgbBuffer = NULL;
        tjDestroy(jpegDecompressor);
        return NULL;
    }

    free(jpegBuffer);
    jpegBuffer = NULL;
    tjDestroy(jpegDecompressor);

    int*** intArray = calloc_3d_int_array(*width, *height, 3);
    if (!intArray) {
        fprintf(stderr, "Error allocating memory for float array\n");
        free(rgbBuffer);
        rgbBuffer = NULL;
        return NULL;
    }

    int *p = &intArray[0][0][0];
    for (int index = 0; index < *height * *width * 3; ++index) {
        p[index] = (int)rgbBuffer[index];
    }

    free(rgbBuffer);
    rgbBuffer = NULL;

    return intArray;
}

static float*** loadFloatPNG(const char* pngFileName, int* width, int* height) {
    FILE* pngFile = fopen(pngFileName, "rb");
    if (!pngFile) {
        fprintf(stderr, "Error opening PNG file\n");
        return NULL;
    }

    png_structp pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!pngPtr) {
        fprintf(stderr, "Error creating PNG read structure\n");
        fclose(pngFile);
        return NULL;
    }

    png_infop infoPtr = png_create_info_struct(pngPtr);
    if (!infoPtr) {
        fprintf(stderr, "Error creating PNG info structure\n");
        png_destroy_read_struct(&pngPtr, NULL, NULL);
        fclose(pngFile);
        return NULL;
    }

    if (setjmp(png_jmpbuf(pngPtr))) {
        fprintf(stderr, "Error during PNG read\n");
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(pngFile);
        return NULL;
    }

    png_init_io(pngPtr, pngFile);
    png_read_info(pngPtr, infoPtr);

    int bitDepth, colorType;
    png_get_IHDR(pngPtr, infoPtr, (unsigned int*)width, (unsigned int*)height, &bitDepth, &colorType, NULL, NULL, NULL);

    if (bitDepth != 8) {
        fprintf(stderr, "Unsupported bit depth in PNG image (must be 8)\n");
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(pngFile);
        return NULL;
    }

    if (colorType != PNG_COLOR_TYPE_RGB && colorType != PNG_COLOR_TYPE_RGBA) {
        fprintf(stderr, "Unsupported color type in PNG image (must be RGB or RGBA)\n");
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(pngFile);
        return NULL;
    }

    int channels = (colorType == PNG_COLOR_TYPE_RGB) ? 3 : 4;
    *width = png_get_image_width(pngPtr, infoPtr);
    *height = png_get_image_height(pngPtr, infoPtr);

    float*** imageData = calloc_3d_float_array(*width, *height, channels);
    if (!imageData) {
        fprintf(stderr, "Error allocating memory for image data\n");
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(pngFile);
        return NULL;
    }

    png_bytep rowBuffer = (png_bytep)malloc(png_get_rowbytes(pngPtr, infoPtr));
    if (!rowBuffer) {
        fprintf(stderr, "Error allocating memory for row buffer\n");
        for (int i = 0; i < *height; ++i) {
            for (int j = 0; j < *width; ++j) {
                free(imageData[i][j]);
                imageData[i][j] = NULL;
            }
            free(imageData[i]);
            imageData[i] = NULL;
        }
        free(imageData);
        imageData = NULL;
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(pngFile);
        return NULL;
    }

    for (int y = 0; y < *height; ++y) {
        png_read_row(pngPtr, rowBuffer, NULL);
        for (int x = 0; x < *width; ++x) {
            for (int c = 0; c < channels; ++c) {
                imageData[y][x][c] = rowBuffer[x * channels + c];
            }
        }
    }

    png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
    fclose(pngFile);
    free(rowBuffer);
    rowBuffer = NULL;

    return imageData;
}

static int*** loadIntPNG(const char* pngFileName, int* width, int* height) {
    FILE* pngFile = fopen(pngFileName, "rb");
    if (!pngFile) {
        fprintf(stderr, "Error opening PNG file\n");
        return NULL;
    }

    png_structp pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!pngPtr) {
        fprintf(stderr, "Error creating PNG read structure\n");
        fclose(pngFile);
        return NULL;
    }

    png_infop infoPtr = png_create_info_struct(pngPtr);
    if (!infoPtr) {
        fprintf(stderr, "Error creating PNG info structure\n");
        png_destroy_read_struct(&pngPtr, NULL, NULL);
        fclose(pngFile);
        return NULL;
    }

    if (setjmp(png_jmpbuf(pngPtr))) {
        fprintf(stderr, "Error during PNG read\n");
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(pngFile);
        return NULL;
    }

    png_init_io(pngPtr, pngFile);
    png_read_info(pngPtr, infoPtr);

    int bitDepth, colorType;
    png_get_IHDR(pngPtr, infoPtr, (unsigned int*)width, (unsigned int*)height, &bitDepth, &colorType, NULL, NULL, NULL);

    if (bitDepth != 8) {
        fprintf(stderr, "Unsupported bit depth in PNG image (must be 8)\n");
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(pngFile);
        return NULL;
    }

    if (colorType != PNG_COLOR_TYPE_RGB && colorType != PNG_COLOR_TYPE_RGBA) {
        fprintf(stderr, "Unsupported color type in PNG image (must be RGB or RGBA)\n");
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(pngFile);
        return NULL;
    }

    int channels = (colorType == PNG_COLOR_TYPE_RGB) ? 3 : 4;
    *width = png_get_image_width(pngPtr, infoPtr);
    *height = png_get_image_height(pngPtr, infoPtr);

    int*** imageData = calloc_3d_int_array(*width, *height, channels);
    if (!imageData) {
        fprintf(stderr, "Error allocating memory for image data\n");
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(pngFile);
        return NULL;
    }

    png_bytep rowBuffer = (png_bytep)malloc(png_get_rowbytes(pngPtr, infoPtr));
    if (!rowBuffer) {
        fprintf(stderr, "Error allocating memory for row buffer\n");
        for (int i = 0; i < *height; ++i) {
            for (int j = 0; j < *width; ++j) {
                free(imageData[i][j]);
                imageData[i][j] = NULL;
            }
            free(imageData[i]);
            imageData[i] = NULL;
        }
        free(imageData);
        imageData = NULL;
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(pngFile);
        return NULL;
    }

    for (int y = 0; y < *height; ++y) {
        png_read_row(pngPtr, rowBuffer, NULL);
        for (int x = 0; x < *width; ++x) {
            for (int c = 0; c < channels; ++c) {
                imageData[y][x][c] = rowBuffer[x * channels + c];
            }
        }
    }

    png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
    fclose(pngFile);
    free(rowBuffer);
    rowBuffer = NULL;

    return imageData;
}

static float*** loadFloatImage(const char* fileName, int* width, int* height, int* channels) {
    int desiredChannels = 0; // Keep the original number of channels
    stbi_uc* imageData = stbi_load(fileName, width, height, channels, desiredChannels);
    if (!imageData) {
        fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
        return NULL;
    }

    // Check if the image has an alpha channel
    if (*channels == 4) {
        // If the image has an alpha channel, remove it by copying only the RGB channels
        float*** rgbImageArray = calloc_3d_float_array(*width, *height, 3);
        if (!rgbImageArray) {
            fprintf(stderr, "Error allocating memory for image array\n");
            stbi_image_free(imageData);
            return NULL;
        }

        float* p = (float *)rgbImageArray[0][0];
        for (int index = 0; index < *width * *height * 3; ++index) {
            p[index] = (float)imageData[index];
        }

        stbi_image_free(imageData);

        *channels = 3; // Update channels to indicate RGB channels only
        return rgbImageArray;
    }

    // If the image doesn't have an alpha channel, just return the loaded image data
    float*** imageArray = calloc_3d_float_array(*width, *height, *channels);
    if (!imageArray) {
        fprintf(stderr, "Error allocating memory for image array\n");
        stbi_image_free(imageData);
        return NULL;
    }

    float* p = &imageArray[0][0][0];
    for (int index = 0; index < *width * *height * 3; ++index) {
        p[index] = (float)imageData[index];
    }

    stbi_image_free(imageData);

    return imageArray;
}

static int*** loadIntImage(const char* fileName, int* width, int* height, int* channels) {
    int desiredChannels = 0; // Keep the original number of channels
    stbi_uc* imageData = stbi_load(fileName, width, height, channels, desiredChannels);
    if (!imageData) {
        fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
        return NULL;
    }

    // Check if the image has an alpha channel
    if (*channels == 4) {
        // If the image has an alpha channel, remove it by copying only the RGB channels
        int*** rgbImageArray = calloc_3d_int_array(*width, *height, 3);
        if (!rgbImageArray) {
            fprintf(stderr, "Error allocating memory for image array\n");
            stbi_image_free(imageData);
            return NULL;
        }

        int* p = &rgbImageArray[0][0][0];
        for (int index = 0; index < *width * *height * 3; ++index) {
            p[index] = (int)imageData[index];
        }

        stbi_image_free(imageData);

        *channels = 3; // Update channels to indicate RGB channels only
        return rgbImageArray;
    }

    // If the image doesn't have an alpha channel, just return the loaded image data
    int*** imageArray = calloc_3d_int_array(*width, *height, *channels);
    if (!imageArray) {
        fprintf(stderr, "Error allocating memory for image array\n");
        stbi_image_free(imageData);
        return NULL;
    }

    int* p = &imageArray[0][0][0];
    for (int index = 0; index < *width * *height * 3; ++index) {
        p[index] = (int)imageData[index];
    }

    stbi_image_free(imageData);

    return imageArray;
}

// TODO: Load image data in the specified format

InputData* load_image_data_with_format(const char* filename, const Dimensions* input_dimensions, DataType data_type, ImageFormat format) {
    // Create an input data structure
    InputData* image_data = ( InputData* )malloc(sizeof(InputData));
    if (image_data == NULL) {
        // Handle memory allocation error
        fprintf(stderr, "Error allocating memory for image data\n");
        return NULL;
    }

    // Initialize image_data to NULL
    image_data->float32_data = NULL;
//    image_data->int_data = NULL;

    // Determine image dimensions
    Dimensions image_dimensions;
    image_dimensions.channels = input_dimensions->channels;

    // Load image data based on format and data type
    int load_success = 0;
    switch (format) {
        case JPEG:
        case JPG:
            if (data_type == FLOAT32) {
                image_data->float32_data = loadFloatJPEG(filename, &image_dimensions.width, &image_dimensions.height);
                load_success = (image_data->float32_data != NULL);
            }else if (data_type == SINT32) {
                image_data->int_data = loadIntJPEG(filename, &image_dimensions.width, &image_dimensions.height);
                load_success = (image_data->int_data != NULL);
            }
            break;

        case PNG:
            if (data_type == FLOAT32) {
                image_data->float32_data = loadFloatPNG(filename, &image_dimensions.width, &image_dimensions.height);
                load_success = (image_data->float32_data != NULL);
            } else if (data_type == SINT32) {
                image_data->int_data = loadIntPNG(filename, &image_dimensions.width, &image_dimensions.height);
                load_success = (image_data->int_data != NULL);
            }
            break;

        case GIF:
        case TGA:
        case BMP:
        case PSD:
        case HDR:
        case PIC:
            if (data_type == FLOAT32) {
                image_data->float32_data = loadFloatImage(filename, &image_dimensions.width, &image_dimensions.height, &image_dimensions.channels);
                load_success = (image_data->float32_data != NULL);
            } else if (data_type == SINT32) {
                image_data->int_data = loadIntImage(filename, &image_dimensions.width, &image_dimensions.height, &image_dimensions.channels);
                load_success = (image_data->int_data != NULL);
            }
            break;

        default:
            // Handle unsupported image format
            fprintf(stderr, "Unsupported image format\n");
            free(image_data);
            image_data = NULL;
            return NULL;
    }

    if (!load_success) {
        // Handle error loading image data
        if (data_type == FLOAT32) {
            fprintf(stderr, "Error loading image data as float array\n");
        } else if (data_type == SINT32) {
            fprintf(stderr, "Error loading image data as int array\n");
        }
        free(image_data);
        image_data = NULL;
        return NULL;
    }

    // Resize the image data if needed
    if (image_dimensions.width != input_dimensions->width || image_dimensions.height != input_dimensions->height) {
        int is_error = resize_image(&image_data, image_dimensions, *input_dimensions, data_type);
        if (is_error == 0) {
            return image_data;
        } else if (is_error == -1) {
            fprintf(stderr, "Unvalid dimensions for resizing\n");
            free_image_data(image_data, image_dimensions, data_type);
            return NULL;
        } else if (is_error == -2) {
            fprintf(stderr, "Error allocating the resized image memory\n");
            free_image_data(image_data, image_dimensions, data_type);
            return NULL;
        }
    }

    return image_data;
}

InputData* create_empty_input_data(int width, int height, int channels, DataType data_type, int fill_value) {
    // Create an InputData structure
    InputData *image_data = (InputData*)malloc(sizeof(InputData));
    if (image_data == NULL) {
        // Handle memory allocation error
        fprintf(stderr, "Error allocating memory for inputData structure\n");
        return NULL;
    }

    // Initialize pointers to NULL
    image_data->float32_data = NULL;

    // Allocate memory for pixel values based on data type and dimensions
    switch(data_type) {
        case SINT32:
            image_data->int_data = calloc_3d_int_array(width, height, channels);
            if (image_data->int_data == NULL) {
                // Handle memory allocation error
                fprintf(stderr, "Error allocating memory for int data\n");
                free(image_data);
                image_data = NULL;
                return NULL;
            }

            // Optionally initialize pixel values to fill_value
            int* int_p = &image_data->int_data[0][0][0];
            if (fill_value != 0) {
                for (int index = 0; index < width * height * channels; index++) {
                    int_p[index] = fill_value;
                }
            }

            break;
        case FLOAT32:
            image_data->float32_data = calloc_3d_float_array(width, height, channels);
            if (image_data->float32_data == NULL) {
                // Handle memory allocation error
                fprintf(stderr, "Error allocating memory for float32 data\n");
                free(image_data);
                image_data = NULL;
                return NULL;
            }

            // Optionally initialize pixel values to fill_value
            float* float_p = &image_data->float32_data[0][0][0];
            if (fill_value != 0) {
                for (int index = 0; index < width * height * channels; index++) {
                    float_p[index] = (float)fill_value;
                }
            }

            break;
    }

    return image_data;
}

int resize_image(InputData **image_data_ptr, const Dimensions original_dimensions, Dimensions new_dimensions, DataType data_type) {
    InputData *image_data = *image_data_ptr;

    // Check for valid dimensions
    if (original_dimensions.width <= 0 || original_dimensions.height <= 0 || new_dimensions.width <= 0 || new_dimensions.height <= 0) {
        return -1; // Invalid dimensions
    }

    // Calculate scaling factors
    float scale_x = (float)new_dimensions.width / original_dimensions.width;
    float scale_y = (float)new_dimensions.height / original_dimensions.height;

    // Allocate memory for the resized image
    InputData *resized_image = create_empty_input_data(new_dimensions.width, new_dimensions.height, original_dimensions.channels, data_type, 0);
    if (resized_image == NULL) {
        // Handle memory allocation error
        return -2; // Memory allocation failed
    }

    // Resize the image using interpolation
    for (int y = 0; y < new_dimensions.height; y++) {
        for (int x = 0; x < new_dimensions.width; x++) {
            for (int c = 0; c < original_dimensions.channels; c++) {
                float source_x = ((float)x + 0.5f) / scale_x - 0.5f; // Center the sampling point
                float source_y = ((float)y + 0.5f) / scale_y - 0.5f; // Center the sampling point

                // Perform bilinear interpolation
                int x0 = (int)floorf(source_x);
                int y0 = (int)floorf(source_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                // Clamp the indices to the input data bounds
                x0 = (x0 < 0) ? 0 : x0;
                x1 = (x1 >= original_dimensions.width) ? original_dimensions.width - 1 : x1;
                y0 = (y0 < 0) ? 0 : y0;
                y1 = (y1 >= original_dimensions.height) ? original_dimensions.height - 1 : y1;

                float dx = source_x - x0;
                float dy = source_y - y0;

                // Calculate the interpolated value based on data type
                if (data_type == SINT32) {
                    int top_left = image_data->int_data[y0][x0][c];
                    int top_right = image_data->int_data[y0][x1][c];
                    int bottom_left = image_data->int_data[y1][x0][c];
                    int bottom_right = image_data->int_data[y1][x1][c];

                    int interpolated_value = (int)(top_left * (1 - dx) * (1 - dy) + top_right * dx * (1 - dy) +
                                                   bottom_left * (1 - dx) * dy + bottom_right * dx * dy + 0.5f);

                    resized_image->int_data[y][x][c] = interpolated_value;
                } else if (data_type == FLOAT32) {
                    float top_left = image_data->float32_data[y0][x0][c];
                    float top_right = image_data->float32_data[y0][x1][c];
                    float bottom_left = image_data->float32_data[y1][x0][c];
                    float bottom_right = image_data->float32_data[y1][x1][c];

                    float interpolated_value = top_left * (1 - dx) * (1 - dy) + top_right * dx * (1 - dy) +
                                                bottom_left * (1 - dx) * dy + bottom_right * dx * dy;

                    resized_image->float32_data[y][x][c] = interpolated_value;
                }
            }
        }
    }

    free_image_data(*image_data_ptr, original_dimensions, data_type);
    *image_data_ptr = resized_image;
    return 0; // Success
}

InputData* copy_image_data(InputData* src, Dimensions dimensions, DataType data_type) {
    if (src == NULL) {
        return NULL;
    }

    InputData* dst = (InputData*)malloc(sizeof(InputData));
    if (dst == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    switch (data_type) {
        case SINT32:
            dst->int_data = copy_3d_int_array(src->int_data, dimensions.width, dimensions.height, dimensions.channels);
            if (dst->int_data == NULL) {
                free(dst);
                fprintf(stderr, "Error: Memory allocation failed\n");
                return NULL;
            }
            break;
        case FLOAT32:
            dst->float32_data = copy_3d_float_array(src->float32_data, dimensions.width, dimensions.height, dimensions.channels);
            if (dst->float32_data == NULL) {
                free(dst);
                fprintf(stderr, "Error: Memory allocation failed\n");
                return NULL;
            }
            break;
    }

    return dst;
}

void free_image_data(InputData *image_data, Dimensions dimensions, DataType data_type) {
    if (image_data == NULL) {
        return;
    }

    // Free allocated memory for pixel values
    switch(data_type) {
        case SINT32:
            if (image_data->int_data != NULL) {
                free_3d_int_array(image_data->int_data);
                image_data->int_data = NULL;
            }
            break;
        case FLOAT32:
            if (image_data->float32_data != NULL) {
                free_3d_float_array(image_data->float32_data);
                image_data->float32_data = NULL;
            }
            break;
    }

    // Free InputData structure itself
    free(image_data);
    image_data = NULL;
}
