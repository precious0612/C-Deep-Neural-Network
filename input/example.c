/* input/example.c */

#include <stdio.h>

#include "data.h"

int main() {
    // Load the test image
    const char *filename = "input/test_pic/0/test.jpeg";
    Dimensions new_dimensions;
    new_dimensions.width = 100;
    new_dimensions.height = 100;
    new_dimensions.channels = 3;
    InputData *image_data = load_input_data_from_image(filename, &new_dimensions, Int);
    if (image_data == NULL) {
        fprintf(stderr, "Error: Failed to load the image.\n");
        return 1;
    }

    // print the first two line
    for (int i = 66; i < 70; i++) {
        for (int j = 88; j < 91; j++) {
            printf("%d ", image_data->int_data[i][j][0]);
        }
        printf("\n");
    }

    // Free the memory allocated for image data
    free_image_data(image_data, new_dimensions, Int);

    printf("Image processing completed successfully.\n");
    return 0;
}