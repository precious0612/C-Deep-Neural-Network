//
//  optim_algorithm.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/20/24.
//

#include "optim_algorithm.h"

#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <math.h>

void sgd(float* weights, float* weight_grads, float momentum, float* momentum_buffer, float learning_rate, int length) {
    
    if (weights == NULL || weight_grads == NULL || momentum_buffer == NULL) {
        fprintf(stderr, "Optimizer input is NULL\n");
        return;
    }
    
    cblas_saxpby(length, learning_rate, weight_grads, 1, momentum, momentum_buffer, 1);  // momentum_buffer = momentum * momentum_buffer + learning_rate * weight_grads
    cblas_saxpy(length, -1.0f, momentum_buffer, 1, weights, 1);                          // weights         = weights - momentum_buffer
}

void adam(float* weights, float* weight_grads, float* m, float* v, float beta1, float beta2, float epsilon, int t, float learning_rate, int length) {
    
    if (weights == NULL || weight_grads == NULL || m == NULL || v == NULL) {
        fprintf(stderr, "Optimizer input is NULL\n");
        return;
    }
    
    float* temp1 = (float *)calloc(length, sizeof(float));
    if (temp1 == NULL) {
        fprintf(stderr, "Error allocating memory during updating weights\n");
        return;
    }
    float* temp2 = (float *)calloc(length, sizeof(float));
    if (temp2 == NULL) {
        fprintf(stderr, "Error allocating memory during updating weights\n");
        free(temp1);
        temp1 = NULL;
        return;
    }
    for (int i = 0; i < length; ++i) {
        temp1[i] = powf(weight_grads[i], 2.0f);                                             // temp1   = weight_grads * weight_grads
    }
    cblas_saxpby(length, 1.0f - beta1, weight_grads, 1, beta1, m, 1);                       // m       = beta1 * m + (1 - beta1) * weight_grads
    cblas_saxpby(length, 1.0f - beta2, temp1, 1, beta2, v, 1);                              // v       = beta2 * v + (1 - beta2) * temp1
    cblas_saxpby(length, powf(1.0f - powf(beta1, (float)t), -1.0f), m, 1, 0.0f, temp1, 1);  // temp1   = m * (1 - beta1^2)^(-1)
    cblas_saxpby(length, powf(1.0f - powf(beta2, (float)t), -1.0f), v, 1, 0.0f, temp2, 1);  // temp2   = m * (1 - beta1^2)^(-1)
    for (int i = 0; i < length; ++i) {
        temp1[i] *= powf(sqrtf(temp2[i]) - epsilon, -1.0f);                                 // temp1   = temp1 * (sqrt(temp2) + epsilon)^(-1)
    }
    cblas_saxpby(length, 0.0f - learning_rate, temp1, 1, 1.0f, weights, 1);                 // weights = weights - learning_rate * temp1
    free(temp1);
    temp1 = NULL;
    free(temp2);
    temp2 = NULL;
}

void rmsprop(float* weights, float* weight_grads, float* square_avg_grad, float rho, float epsilon, float learning_rate, int length) {
    
    if (weights == NULL || weight_grads == NULL || square_avg_grad == NULL) {
        fprintf(stderr, "Optimizer input is NULL\n");
        return;
    }
    
    float* temp = (float *)calloc(length, sizeof(float));
    if (temp == NULL) {
        fprintf(stderr, "Error allocating memory during updating weights\n");
        return;
    }
    for (int i = 0; i < length; ++i) {
        temp[i] = powf(weight_grads[i], 2.0f);                                         // temp            = weight_grads * weight_grads
    }
    cblas_saxpby(length, 1.0f - rho, temp, 1, rho, square_avg_grad, 1);                // square_avg_grad = rho * square_avg_grad + (1 - rho) * temp
    for (int i = 0; i < length; ++i) {
        temp[i] = weight_grads[i] * powf(sqrtf(square_avg_grad[i]) + epsilon, -1.0f);  // temp            = weight_grads * (sqrt(square_avg_grad) + epsilon)^(-1)
    }
    cblas_saxpby(length, learning_rate, temp, 1, 1.0f, weights, 1);                    // weights         = weights - learning_rate * temp
    free(temp);
    temp = NULL;
}
