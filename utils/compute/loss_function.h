//
//  loss_function.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/24/24.
//

#ifndef loss_function_h
#define loss_function_h

float categorical_crossentropy_loss(float* output, int label, int num_classes);

float mean_squared_error_loss(float* output, int label, int num_classes);

#endif /* loss_function_h */
