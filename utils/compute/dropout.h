//
//  dropout.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/20/24.
//

#ifndef dropout_h
#define dropout_h

/// Performs the forward pass through a dropout layer.
/// - Parameters:
///   - input: A 3D array containing the input data.
///   - input_width: The width of the input data.
///   - input_height: The height of the input data.
///   - input_channels: The channels of the input data.
///   - dropout_rate: The fraction of input units to set to zero.
/// - Returns: A 3D array containing the output data after applying dropout.
///
/// - Example Usage:
///     ```c
///     float*** output_data = dropout_forward(<input_data>, 3, 3, 1, <dropout_rate>);
///     ```
///
float*** dropout_forward(float*** input, int input_width, int input_height, int input_channels, float dropout_rate);

/// Performs the backward pass through a dropout layer.
/// - Parameters:
///   - input: A 3D array containing the input data used during the forward pass.
///   - output_grad: A 3D array containing the gradients of the output with respect to the loss.
///   - input_grad: A 3D array to store the gradients of the input with respect to the loss.
///   - input_size: The number of the input data.
///
/// - Example Usage:
///     ```c
///     dropout_backward(<input_data>, <output_grad>, <input_grad>, 9);
///     ```
///     
void dropout_backward(float*** input, float*** output_grad, float*** input_grad, int input_size);

#endif /* dropout_h */
