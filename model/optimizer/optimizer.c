//
//  optimizer.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/18/24.
//

#include "optimizer.h"

#include <stdio.h>
#include <stdlib.h>

static void free_sgd(SGDOptimizer* sgd, int num_layers) {
    if (sgd == NULL) {
        return;
    }

    if (sgd->momentum_buffer != NULL) {
        for (int i = 0; i < num_layers; ++i) {
            if (sgd->momentum_buffer[i] != NULL) {
                free(sgd->momentum_buffer[i]);
                sgd->momentum_buffer[i] = NULL;
            }
        }
        free(sgd->momentum_buffer);
        sgd->momentum_buffer = NULL;
    }

    free(sgd);
    sgd = NULL;
}

static SGDOptimizer* init_sgd(LearningRate learning_rate, float momentum, int* num_weights, int num_layers) {
    
    if (learning_rate <= 0.0f || momentum < 0.0f || momentum > 1.0f || num_weights == NULL || num_layers <= 0) {
            fprintf(stderr, "Error: Invalid input parameters\n");
            return NULL;
        }
    
    SGDOptimizer* sgd = (SGDOptimizer*)malloc(sizeof(SGDOptimizer));
    if (sgd == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for SGD optimizer\n");
        return NULL;
    }

    sgd->learning_rate = learning_rate;
    sgd->momentum      = momentum;

    sgd->momentum_buffer = (float**)malloc(num_layers * sizeof(float*));
    if (sgd->momentum_buffer == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for SGD optimizer momentum_buffer\n");
        free(sgd);
        sgd = NULL;
        return NULL;
    }
    
    for (int i = 0; i < num_layers; ++i) {
        sgd->momentum_buffer[i] = (float*)calloc(num_weights[i], sizeof(float));
        if (sgd->momentum_buffer[i] == NULL) {
            fprintf(stderr, "Error: Unable to allocate memory for SGD optimizer momentum_buffer[%d]\n", i);
            free_sgd(sgd, num_layers);
            return NULL;
        }
    }

    return sgd;
}

static void free_adam(AdamOptimizer* adam, int num_layers) {
    if (adam == NULL) {
        return;
    }

    if (adam->m != NULL) {
        for (int i = 0; i < num_layers; ++i) {
            if (adam->m[i] != NULL) {
                free(adam->m[i]);
                adam->m[i] = NULL;
            }
        }
        free(adam->m);
        adam->m = NULL;
    }

    if (adam->v != NULL) {
        for (int i = 0; i < num_layers; ++i) {
            if (adam->v[i] != NULL) {
                free(adam->v[i]);
                adam->v[i] = NULL;
            }
        }
        free(adam->v);
        adam->v = NULL;
    }

    free(adam);
    adam = NULL;
}

static AdamOptimizer* init_adam(LearningRate learning_rate, float beta1, float beta2, float epsilon, int* num_weights, int num_layers) {
    
    if (learning_rate <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || epsilon <= 0.0f || num_weights == NULL || num_layers <= 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return NULL;
    }
    
    AdamOptimizer* adam = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    if (adam == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for Adam optimizer\n");
        free(adam);
        adam = NULL;
        return NULL;
    }

    adam->learning_rate = learning_rate;
    adam->beta1         = beta1;
    adam->beta2         = beta2;
    adam->epsilon       = epsilon;
    adam->t             = 1;

    adam->m = (float**)malloc(num_layers * sizeof(float*));
    if (adam->m == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for Adam optimizer first moment vector\n");
        free(adam);
        adam = NULL;
        return NULL;
    }

    adam->v = (float**)malloc(num_layers * sizeof(float*));
    if (adam->v == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for Adam optimizer second moment vector\n");
        free_adam(adam, num_layers);
    }

    for (int i = 0; i < num_layers; i++) {
        adam->m[i] = (float*)calloc(num_weights[i], sizeof(float));
        if (adam->m[i] == NULL) {
            fprintf(stderr, "Error: Unable to allocate memory for Adam optimizer first moment vector[%d]\n", i);
            free_adam(adam, num_layers);
            return NULL;
        }
        adam->v[i] = (float*)calloc(num_weights[i], sizeof(float));
        if (adam->v[i] == NULL) {
            fprintf(stderr, "Error: Unable to allocate memory for Adam optimizer second moment vector[%d]\n", i);
            free_adam(adam, num_layers);
            return NULL;
        }
    }

    return adam;
}

static void free_rmsprop(RMSpropOptimizer* rmsprop, int num_layers) {
    if (rmsprop == NULL) {
        return;
    }

    if (rmsprop->square_avg_grad != NULL) {
        for (int i = 0; i < num_layers; ++i) {
            if (rmsprop->square_avg_grad[i] != NULL) {
                free(rmsprop->square_avg_grad[i]);
                rmsprop->square_avg_grad[i] = NULL;
            }
        }
        free(rmsprop->square_avg_grad);
        rmsprop->square_avg_grad = NULL;
    }

    free(rmsprop);
    rmsprop = NULL;
}

static RMSpropOptimizer* init_rmsprop(LearningRate learning_rate, float rho, float epsilon, int* num_weights, int num_layers) {
    
    if (learning_rate <= 0.0f || rho < 0.0f || rho >= 1.0f || epsilon <= 0.0f || num_weights == NULL || num_layers <= 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return NULL;
    }

    RMSpropOptimizer* rmsprop = (RMSpropOptimizer*)malloc(sizeof(RMSpropOptimizer));
    if (rmsprop == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for RMSprop optimizer\n");
        return NULL;
    }

    rmsprop->learning_rate = learning_rate;
    rmsprop->rho           = rho;
    rmsprop->epsilon       = epsilon;

    rmsprop->square_avg_grad = (float**)malloc(num_layers * sizeof(float*));
    if (rmsprop->square_avg_grad == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for RMSprop optimizer squared average gradient\n");
        free(rmsprop);
        return NULL;
    }

    for (int i = 0; i < num_layers; i++) {
        rmsprop->square_avg_grad[i] = (float*)calloc(num_weights[i], sizeof(float));
        if (rmsprop->square_avg_grad[i] == NULL) {
            free_rmsprop(rmsprop, num_layers);
            return NULL;
        }
    }

    return rmsprop;
}

Optimizer* create_optimizer(OptimizerType optimizer_type, LearningRate learning_rate, int* num_weights, int num_layers) {
    
    if (learning_rate <= 0.0f) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return NULL;
    }

    Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    if (optimizer == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for optimizer\n");
        return NULL;
    }
    optimizer->type = optimizer_type;
    
    switch (optimizer_type) {
        case SGD:
            optimizer->optimizer.sgd = init_sgd(learning_rate, 0.0f, num_weights, num_layers);
            return optimizer;
        case ADAM:
            optimizer->optimizer.adam = init_adam(learning_rate, 0.9f, 0.999f, 1e-8f, num_weights, num_layers);
            return optimizer;
        case RMSPROP:
            optimizer->optimizer.rmsprop = init_rmsprop(learning_rate, 0.9f, 1e-8f, num_weights, num_layers);
            return optimizer;
        default:
            fprintf(stderr, "Error: Invalid optimizer type\n");
            free(optimizer);
            optimizer = NULL;
            return NULL;
    }
}

void delete_optimizer(Optimizer* optimizer, int num_layers) {
    if (optimizer == NULL) {
        return;
    }

    switch (optimizer->type) {
        case SGD:
            free_sgd(optimizer->optimizer.sgd, num_layers);
            break;
        case ADAM:
            free_adam(optimizer->optimizer.adam, num_layers);
            break;
        case RMSPROP:
            free_rmsprop(optimizer->optimizer.rmsprop, num_layers);
            break;
        default:
            fprintf(stderr, "Error: Invalid optimizer type\n");
            break;
    }

    free(optimizer);
    optimizer = NULL;
}
