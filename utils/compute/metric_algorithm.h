//
//  metric_algorithm.h
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/26/24.
//

#ifndef metric_algorithm_h
#define metric_algorithm_h

int get_prediction(float* output, int num_classes);

float f1_score(float* output, int label, int num_classes);

#endif /* metric_algorithm_h */
