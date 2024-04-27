/* input/input.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <turbojpeg.h> // libjpeg-turbo
#include <png.h> // libpng
// stb_image for other formats
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#include "data.h"

InputData *load_input_data_from_image(const char *filename, const Dimensions *input_dimensions, DataType data_type) {
    // Determine image format based on filename extension
    const char *ext = strrchr(filename, '.');
    ImageFormat format;
    if (ext == NULL) {
        // Unable to determine format, default to JPEG2
        format = PIC;
    } else {
        ext++; // Skip the dot
        if (strcmp(ext, "jpg") == 0 || strcmp(ext, "jpeg") == 0) {
            format = JPEG;
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
            // Default to JPEG for unknown formats
            format = PIC;
        }
    }

    // Delegate to the original load_image_data function with the determined format and provided input dimensions
    return load_image_data_with_format(filename, input_dimensions, data_type, format);
}

InputData *load_image_data_with_format(const char *filename, const Dimensions *input_dimensions, DataType data_type, ImageFormat format) {

    // Create an input data structure
    InputData *image_data = (InputData *)malloc(sizeof(InputData));
    if (image_data == NULL) {
        // Handle memory allocation error
        fprintf(stderr, "Error allocating memory for image data\n");
        return NULL;
    }

    // Determine image dimensions
    Dimensions image_dimensions;
    image_dimensions.channels = input_dimensions->channels;

    // Load image data based on format and data type
    switch (format) {
        case JPEG:
        case JPG:
            if (data_type == FLOAT32) {
                image_data->float32_data = loadFloatJPEG(filename, &image_dimensions.width, &image_dimensions.height);
                resize_image(&image_data, image_dimensions, *input_dimensions, data_type);
            } else if (data_type == Int) {
                image_data->int_data = loadIntJPEG(filename, &image_dimensions.width, &image_dimensions.height);
                resize_image(&image_data, image_dimensions, *input_dimensions, data_type);
            }
            break;

        case PNG:
            if (data_type == FLOAT32) {
                image_data->float32_data = loadFloatPNG(filename, &image_dimensions.width, &image_dimensions.height);
                resize_image(&image_data, image_dimensions, *input_dimensions, data_type);
            } else if (data_type == Int) {
                image_data->int_data = loadIntPNG(filename, &image_dimensions.width, &image_dimensions.height);
                resize_image(&image_data, image_dimensions, *input_dimensions, data_type);
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
                resize_image(&image_data, image_dimensions, *input_dimensions, data_type);
            } else if (data_type == Int) {
                image_data->int_data = loadIntImage(filename, &image_dimensions.width, &image_dimensions.height, &image_dimensions.channels);
                resize_image(&image_data, image_dimensions, *input_dimensions, data_type);
            }
            break;
    }

    if (data_type == FLOAT32 && image_data->float32_data == NULL) {
        // Handle error loading image data as float array
        fprintf(stderr, "Error loading image data as float array\n");
        free(image_data);
        return NULL;
    }

    if (data_type == Int && image_data->int_data == NULL) {
        // Handle error loading image data as int array
        fprintf(stderr, "Error loading image data as int array\n");
        free(image_data);
        return NULL;
    }

    return image_data;
}


float*** loadFloatJPEG(const char* jpegFileName, int* width, int* height) {
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
    fread(jpegBuffer, 1, jpegFileSize, jpegFile);
    fclose(jpegFile);

    int jpegSubsamp;
    tjDecompressHeader2(jpegDecompressor, jpegBuffer, jpegFileSize, width, height, &jpegSubsamp);
    int pixelFormat = TJPF_RGB;
    unsigned char* rgbBuffer = (unsigned char*)malloc((*width) * (*height) * tjPixelSize[pixelFormat]);
    tjDecompress2(jpegDecompressor, jpegBuffer, jpegFileSize, rgbBuffer, *width, 0, *height, pixelFormat, 0);

    float*** floatArray = (float***)malloc((*height) * sizeof(float**));
    for (int i = 0; i < *height; ++i) {
        floatArray[i] = (float**)malloc((*width) * sizeof(float*));
        for (int j = 0; j < *width; ++j) {
            floatArray[i][j] = (float*)malloc(3 * sizeof(float));
            for (int k = 0; k < 3; ++k) {
                floatArray[i][j][k] = (float)rgbBuffer[(i * (*width) + j) * 3 + k];
            }
        }
    }

    free(jpegBuffer);
    free(rgbBuffer);
    tjDestroy(jpegDecompressor);

    return floatArray;
}

int*** loadIntJPEG(const char* jpegFileName, int* width, int* height) {
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
    fread(jpegBuffer, 1, jpegFileSize, jpegFile);
    fclose(jpegFile);

    int jpegSubsamp;
    tjDecompressHeader2(jpegDecompressor, jpegBuffer, jpegFileSize, width, height, &jpegSubsamp);
    int pixelFormat = TJPF_RGB;
    unsigned char* rgbBuffer = (unsigned char*)malloc((*width) * (*height) * tjPixelSize[pixelFormat]);
    tjDecompress2(jpegDecompressor, jpegBuffer, jpegFileSize, rgbBuffer, *width, 0, *height, pixelFormat, 0);

    int*** intArray = (int***)malloc((*height) * sizeof(int**));
    for (int i = 0; i < *height; ++i) {
        intArray[i] = (int**)malloc((*width) * sizeof(int*));
        for (int j = 0; j < *width; ++j) {
            intArray[i][j] = (int*)malloc(3 * sizeof(int));
            for (int k = 0; k < 3; ++k) {
                intArray[i][j][k] = (int)rgbBuffer[(i * (*width) + j) * 3 + k];
            }
        }
    }

    free(jpegBuffer);
    free(rgbBuffer);
    tjDestroy(jpegDecompressor);

    return intArray;
}

float*** loadFloatPNG(const char* pngFileName, int* width, int* height) {
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

    float*** imageData = (float***)malloc(*height * sizeof(float**));
    for (int i = 0; i < *height; ++i) {
        imageData[i] = (float**)malloc(*width * sizeof(float*));
        for (int j = 0; j < *width; ++j) {
            imageData[i][j] = (float*)malloc(channels * sizeof(float));
        }
    }

    png_bytep rowBuffer = (png_bytep)malloc(png_get_rowbytes(pngPtr, infoPtr));
    for (int i = 0; i < *height; ++i) {
        png_read_row(pngPtr, rowBuffer, NULL);
        for (int j = 0; j < *width; ++j) {
            for (int k = 0; k < channels; ++k) {
                imageData[i][j][k] = rowBuffer[j * channels + k];
            }
        }
    }

    png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
    fclose(pngFile);
    free(rowBuffer);

    return imageData;
}

int*** loadIntPNG(const char* pngFileName, int* width, int* height) {
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

    int*** imageData = (int***)malloc(*height * sizeof(int**));
    for (int i = 0; i < *height; ++i) {
        imageData[i] = (int**)malloc(*width * sizeof(int*));
        for (int j = 0; j < *width; ++j) {
            imageData[i][j] = (int*)malloc(channels * sizeof(int));
        }
    }

    png_bytep rowBuffer = (png_bytep)malloc(png_get_rowbytes(pngPtr, infoPtr));
    for (int i = 0; i < *height; ++i) {
        png_read_row(pngPtr, rowBuffer, NULL);
        for (int j = 0; j < *width; ++j) {
            for (int k = 0; k < channels; ++k) {
                imageData[i][j][k] = rowBuffer[j * channels + k];
            }
        }
    }

    png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
    fclose(pngFile);
    free(rowBuffer);

    return imageData;
}

float*** loadFloatImage(const char* fileName, int* width, int* height, int* channels) {
    int desiredChannels = 0; // Keep the original number of channels
    stbi_uc* imageData = stbi_load(fileName, width, height, channels, desiredChannels);
    if (!imageData) {
        fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
        return NULL;
    }

    // Check if the image has an alpha channel
    if (*channels == 4) {
        // If the image has an alpha channel, remove it by copying only the RGB channels
        float*** rgbImageArray = (float***)malloc(*height * sizeof(float**));
        if (!rgbImageArray) {
            fprintf(stderr, "Error allocating memory for image array\n");
            stbi_image_free(imageData);
            return NULL;
        }

        int pixelSize = 3 * sizeof(float); // RGB channels only
        for (int i = 0; i < *height; ++i) {
            rgbImageArray[i] = (float**)malloc(*width * sizeof(float*));
            if (!rgbImageArray[i]) {
                fprintf(stderr, "Error allocating memory for row %d\n", i);
                // Free memory for previously allocated rows
                for (int j = 0; j < i; ++j) {
                    free(rgbImageArray[j]);
                }
                free(rgbImageArray);
                stbi_image_free(imageData);
                return NULL;
            }
            for (int j = 0; j < *width; ++j) {
                rgbImageArray[i][j] = (float*)malloc(pixelSize);
                if (!rgbImageArray[i][j]) {
                    fprintf(stderr, "Error allocating memory for pixel at (%d, %d)\n", i, j);
                    // Free memory for previously allocated pixels in this row
                    for (int k = 0; k < j; ++k) {
                        free(rgbImageArray[i][k]);
                    }
                    // Free memory for this row
                    free(rgbImageArray[i]);
                    // Free memory for previously allocated rows
                    for (int k = 0; k < i; ++k) {
                        free(rgbImageArray[k]);
                    }
                    free(rgbImageArray);
                    stbi_image_free(imageData);
                    return NULL;
                }
                // Copy RGB data from stb_image result, skipping the alpha channel
                for (int n = 0; n < 3; n++) {
                    rgbImageArray[i][j][n] = (float)imageData[(i * (*width) + j) * (*channels) + n];
                }
            }
        }

        stbi_image_free(imageData);

        *channels = 3; // Update channels to indicate RGB channels only
        return rgbImageArray;
    }

    // If the image doesn't have an alpha channel, just return the loaded image data
    float*** imageArray = (float***)malloc(*height * sizeof(float**));
    if (!imageArray) {
        fprintf(stderr, "Error allocating memory for image array\n");
        stbi_image_free(imageData);
        return NULL;
    }

    int pixelSize = *channels * sizeof(float);
    for (int i = 0; i < *height; ++i) {
        imageArray[i] = (float**)malloc(*width * sizeof(float*));
        if (!imageArray[i]) {
            fprintf(stderr, "Error allocating memory for row %d\n", i);
            // Free memory for previously allocated rows
            for (int j = 0; j < i; ++j) {
                free(imageArray[j]);
            }
            free(imageArray);
            stbi_image_free(imageData);
            return NULL;
        }
        for (int j = 0; j < *width; ++j) {
            imageArray[i][j] = (float*)malloc(pixelSize);
            if (!imageArray[i][j]) {
                fprintf(stderr, "Error allocating memory for pixel at (%d, %d)\n", i, j);
                // Free memory for previously allocated pixels in this row
                for (int k = 0; k < j; ++k) {
                    free(imageArray[i][k]);
                }
                // Free memory for this row
                free(imageArray[i]);
                // Free memory for previously allocated rows
                for (int k = 0; k < i; ++k) {
                    free(imageArray[k]);
                }
                free(imageArray);
                stbi_image_free(imageData);
                return NULL;
            }
            // Copy pixel data from stb_image result
            for (int n = 0; n < 3; n++) {
                imageArray[i][j][n] = (float)imageData[(i * (*width) + j) * (*channels) + n];
            }
        }
    }

    stbi_image_free(imageData);

    return imageArray;
}

int*** loadIntImage(const char* fileName, int* width, int* height, int* channels) {
    int desiredChannels = 0; // Keep the original number of channels
    stbi_uc* imageData = stbi_load(fileName, width, height, channels, desiredChannels);
    if (!imageData) {
        fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
        return NULL;
    }

    // Check if the image has an alpha channel
    if (*channels == 4) {
        // If the image has an alpha channel, remove it by copying only the RGB channels
        int*** rgbImageArray = (int***)malloc(*height * sizeof(int**));
        if (!rgbImageArray) {
            fprintf(stderr, "Error allocating memory for image array\n");
            stbi_image_free(imageData);
            return NULL;
        }

        int pixelSize = 3 * sizeof(int); // RGB channels only
        for (int i = 0; i < *height; ++i) {
            rgbImageArray[i] = (int**)malloc(*width * sizeof(int*));
            if (!rgbImageArray[i]) {
                fprintf(stderr, "Error allocating memory for row %d\n", i);
                // Free memory for previously allocated rows
                for (int j = 0; j < i; ++j) {
                    free(rgbImageArray[j]);
                }
                free(rgbImageArray);
                stbi_image_free(imageData);
                return NULL;
            }
            for (int j = 0; j < *width; ++j) {
                rgbImageArray[i][j] = (int*)malloc(pixelSize);
                if (!rgbImageArray[i][j]) {
                    fprintf(stderr, "Error allocating memory for pixel at (%d, %d)\n", i, j);
                    // Free memory for previously allocated pixels in this row
                    for (int k = 0; k < j; ++k) {
                        free(rgbImageArray[i][k]);
                    }
                    // Free memory for this row
                    free(rgbImageArray[i]);
                    // Free memory for previously allocated rows
                    for (int k = 0; k < i; ++k) {
                        free(rgbImageArray[k]);
                    }
                    free(rgbImageArray);
                    stbi_image_free(imageData);
                    return NULL;
                }
                // Copy RGB data from stb_image result, skipping the alpha channel
                for (int n = 0; n < 3; n++) {
                    rgbImageArray[i][j][n] = (int)imageData[(i * (*width) + j) * (*channels) + n];
                }
            }
        }

        stbi_image_free(imageData);

        *channels = 3; // Update channels to indicate RGB channels only
        return rgbImageArray;
    }

    // If the image doesn't have an alpha channel, just return the loaded image data
    int*** imageArray = (int***)malloc(*height * sizeof(int**));
    if (!imageArray) {
        fprintf(stderr, "Error allocating memory for image array\n");
        stbi_image_free(imageData);
        return NULL;
    }

    int pixelSize = *channels * sizeof(int);
    for (int i = 0; i < *height; ++i) {
        imageArray[i] = (int**)malloc(*width * sizeof(int*));
        if (!imageArray[i]) {
            fprintf(stderr, "Error allocating memory for row %d\n", i);
            // Free memory for previously allocated rows
            for (int j = 0; j < i; ++j) {
                free(imageArray[j]);
            }
            free(imageArray);
            stbi_image_free(imageData);
            return NULL;
        }
        for (int j = 0; j < *width; ++j) {
            imageArray[i][j] = (int*)malloc(pixelSize);
            if (!imageArray[i][j]) {
                fprintf(stderr, "Error allocating memory for pixel at (%d, %d)\n", i, j);
                // Free memory for previously allocated pixels in this row
                for (int k = 0; k < j; ++k) {
                    free(imageArray[i][k]);
                }
                // Free memory for this row
                free(imageArray[i]);
                // Free memory for previously allocated rows
                for (int k = 0; k < i; ++k) {
                    free(imageArray[k]);
                }
                free(imageArray);
                stbi_image_free(imageData);
                return NULL;
            }
            // Copy pixel data from stb_image result
            for (int n = 0; n < 3; n++) {
                imageArray[i][j][n] = (int)imageData[(i * (*width) + j) * (*channels) + n];
            }
        }
    }

    stbi_image_free(imageData);

    return imageArray;
}

InputData* create_empty_input_data(int width, int height, int channels, DataType data_type, int fill_value) {
    // Create an InputData structure
    InputData *image_data = (InputData*)malloc(sizeof(InputData));
    if (image_data == NULL) {
        // Handle memory allocation error
        return NULL;
    }

    // Allocate memory for pixel values based on data type and dimensions
    switch(data_type) {
        case Int:
            image_data->int_data = (int***)malloc(height * sizeof(int**));
            if (image_data->int_data == NULL) {
                // Handle memory allocation error
                free(image_data);
                return NULL;
            }
            for (int i = 0; i < height; i++) {
                image_data->int_data[i] = (int**)malloc(width * sizeof(int*));
                if (image_data->int_data[i] == NULL) {
                    // Handle memory allocation error
                    free(image_data->int_data);
                    free(image_data);
                    return NULL;
                }
                for (int j = 0; j < width; j++) {
                    image_data->int_data[i][j] = (int*)malloc(channels * sizeof(int));
                    if (image_data->int_data[i][j] == NULL) {
                        // Handle memory allocation error
                        for (int k = 0; k < j; k++) {
                            free(image_data->int_data[i][k]);
                        }
                        free(image_data->int_data[i]);
                        free(image_data->int_data);
                        free(image_data);
                        return NULL;
                    }
                    // Optionally initialize pixel values to fill_value
                    for (int k = 0; k < channels; k++) {
                        image_data->int_data[i][j][k] = (int)fill_value;
                    }
                }
            }
            break;
        case FLOAT32:
            image_data->float32_data = (float***)malloc(height * sizeof(float**));
            if (image_data->float32_data == NULL) {
                // Handle memory allocation error
                free(image_data);
                return NULL;
            }
            for (int i = 0; i < height; i++) {
                image_data->float32_data[i] = (float**)malloc(width * sizeof(float*));
                if (image_data->float32_data[i] == NULL) {
                    // Handle memory allocation error
                    free(image_data->float32_data);
                    free(image_data);
                    return NULL;
                }
                for (int j = 0; j < width; j++) {
                    image_data->float32_data[i][j] = (float*)malloc(channels * sizeof(float));
                    if (image_data->float32_data[i][j] == NULL) {
                        // Handle memory allocation error
                        for (int k = 0; k < j; k++) {
                            free(image_data->float32_data[i][k]);
                        }
                        free(image_data->float32_data[i]);
                        free(image_data->float32_data);
                        free(image_data);
                        return NULL;
                    }
                    // Optionally initialize pixel values to fill_value
                    for (int k = 0; k < channels; k++) {
                        image_data->float32_data[i][j][k] = (float)fill_value;
                    }
                }
            }
            break;
    }

    return image_data;
}

// Function to resize the image using interpolation
void resize_image(InputData **image_data_ptr, const Dimensions original_dimensions, Dimensions new_dimensions, DataType data_type) {
    InputData *image_data = *image_data_ptr;

    // Calculate scaling factors
    float scale_x = (float)new_dimensions.width / original_dimensions.width;
    float scale_y = (float)new_dimensions.height / original_dimensions.height;

    // Allocate memory for the resized image
    InputData *resized_image = create_empty_input_data(new_dimensions.width, new_dimensions.height, original_dimensions.channels, data_type, 0);
    if (resized_image == NULL) {
        // Handle memory allocation error
        return;
    }

    // Resize the image using interpolation
    for (int y = 0; y < new_dimensions.height; y++) {
        for (int x = 0; x < new_dimensions.width; x++) {
            for (int c = 0; c < original_dimensions.channels; c++) {
                float source_x = x / scale_x;
                float source_y = y / scale_y;

                // Perform bilinear interpolation
                int x0 = (int)source_x;
                int y0 = (int)source_y;
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                float dx = source_x - x0;
                float dy = source_y - y0;

                // Calculate the interpolated value based on data type
                if (data_type == Int) {
                    int top_left = image_data->int_data[y0][x0][c];
                    int top_right = image_data->int_data[y0][x1][c];
                    int bottom_left = image_data->int_data[y1][x0][c];
                    int bottom_right = image_data->int_data[y1][x1][c];

                    int interpolated_value = top_left * (1 - dx) * (1 - dy) + top_right * dx * (1 - dy) +
                                             bottom_left * (1 - dx) * dy + bottom_right * dx * dy;

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
}

// Preprocess input data (e.g., normalization, resizing)
// void preprocess_input(InputData* input_data, Dimensions input_shape) {
//     // Assuming input_data is already loaded and preprocessed
// }

void free_image_data(InputData *image_data, Dimensions dimensions, DataType data_type) {
    if (image_data == NULL) {
        return;
    }

    // Free allocated memory for pixel values
    switch(data_type) {
        case Int:
            for (int i = 0; i < dimensions.height; i++) {
                for (int j = 0; j < dimensions.width; j++) {
                    free(image_data->int_data[i][j]);
                }
                free(image_data->int_data[i]);
            }
            free(image_data->int_data);
            break;
        case FLOAT32:
            for (int i = 0; i < dimensions.height; i++) {
                for (int j = 0; j < dimensions.width; j++) {
                    free(image_data->float32_data[i][j]);
                }
                free(image_data->float32_data[i]);
            }
            free(image_data->float32_data);
            break;
    }

    // Free InputData structure itself
    free(image_data);
}

