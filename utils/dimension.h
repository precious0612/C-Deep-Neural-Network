//
//  dimension.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/15/24.
//
// This file defines a simple data structure, Dimensions, to represent the shape
// of multi-dimensional tensors or arrays commonly used in deep learning applications.
// The Dimensions struct encapsulates the width, height, and number of channels (depth)
// of a tensor, providing a convenient way to store and pass around this information.
//
// This header file serves as a central hub for working with tensor dimensions, ensuring
// consistency and simplifying the handling of tensor shapes throughout the codebase.
// It can be used in conjunction with other modules that operate on tensors, such as
// convolutional layers, pooling layers, or fully connected layers, to specify the
// input and output shapes of these operations.
//
// Key functionalities include:
//
// 1. Defining the Dimensions struct to hold the width, height, and number of channels.
// 2. Providing a consistent interface for representing tensor shapes throughout the codebase.
// 3. Facilitating the passing of tensor dimensions as function parameters or return values.
//
// Usage examples:
//
// Dimensions input_shape = {32, 32, 3}; // Width: 32, Height: 32, Channels: 3
// Dimensions output_shape = {16, 16, 32}; // Width: 16, Height: 16, Channels: 32
//
// float*** input_tensor = allocate_3d_array(input_shape);
// float*** output_tensor = convolutional_layer(input_tensor, input_shape, output_shape, ...);
//

#ifndef dimension_h
#define dimension_h

typedef struct {
    int width;
    int height;
    int channels;
}Dimensions;

#endif /* dimension_h */
