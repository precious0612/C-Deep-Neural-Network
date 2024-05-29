//
//  losses.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/24/24.
//

#include "losses.h"

#include <stdlib.h>
#include <stdio.h>

#include "../../utils/utils.h"

LossFunction* init_loss_function(LossType type) {
    
    LossFunction* loss = (LossFunction*)malloc(sizeof(LossFunction));
    if (loss == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for loss function\n");
        return NULL;
    }
    
    loss->type = type;
    switch (type) {
        case CrossEntropy:
            loss->loss_function = categorical_crossentropy_loss;
            break;
            
        case MSE:
            loss->loss_function = mean_squared_error_loss;
            break;
            
        default:
            fprintf(stderr, "Error: Invalid loss function type\n");
            free(loss);
            loss = NULL;
            return NULL;
    }
    
    return loss;
    
}

void delete_loss_function(LossFunction* loss) {
    
    if (loss == NULL) {
        return;
    }
    
    free(loss);
    loss = NULL;
}
