//
//  pooling.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/20/24.
//

#ifndef pooling_h
#define pooling_h

typedef enum {
    MAX,
    AVARAGE
} PoolType;

float*** pool_forward(float*** input, int input_width, int input_height, int channels, int pool_size, int stride, PoolType pool_type);

void pool_backward(float*** input, float*** output_grad, float*** input_grad, int input_width, int input_height, int channels, int pool_size, int stride, PoolType pool_type);

#endif /* pooling_h */
