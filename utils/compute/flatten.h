//
//  flatten.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/21/24.
//

#ifndef flatten_h
#define flatten_h

float*** flatten(float*** input, int input_width, int input_height, int input_channles);

float*** unflatten(float*** input, int output_width, int output_height, int output_channles);

#endif /* flatten_h */
