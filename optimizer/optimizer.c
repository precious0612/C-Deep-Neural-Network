/* optimizer/optimizer.c */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "optimizer.h"

SGDOptimizer* init_sgd(float learning_rate, float momentum, int* num_weights, int num_layers) {
    SGDOptimizer* sgd = (SGDOptimizer*)malloc(sizeof(SGDOptimizer));
    if (sgd == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for SGD optimizer\n");
        exit(1);
    }

    sgd->learning_rate = learning_rate;
    sgd->momentum = momentum;

    sgd->momentum_buffer = malloc(num_layers * sizeof(float*));
    if (sgd->momentum_buffer == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for SGD optimizer momentum_buffer\n");
        exit(1);
    }
    for (int i = 0; i < num_layers; i++) {
        sgd->momentum_buffer[i] = calloc(num_weights[i], sizeof(float));
        if (sgd->momentum_buffer[i] == NULL) {
            fprintf(stderr, "Error: Unable to allocate memory for SGD optimizer momentum_buffer[%d]\n", i);
            exit(1);
        }
    }

    return sgd;
}

AdamOptimizer* init_adam(float learning_rate, float beta1, float beta2, float epsilon, int* num_weights, int num_layers) {
    AdamOptimizer* adam = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    if (adam == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for Adam optimizer\n");
        exit(1);
    }

    adam->learning_rate = learning_rate;
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->epsilon = epsilon;
    adam->t = 1;

    adam->m = malloc(num_layers * sizeof(float**));
    if (adam->m == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for Adam optimizer first moment vector\n");
        exit(1);
    }

    adam->v = malloc(num_layers * sizeof(float**));
    if (adam->v == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for Adam optimizer second moment vector\n");
        exit(1);
    }

    for (int i = 0; i < num_layers; i++) {
        adam->m[i] = malloc(num_weights[i] * sizeof(float));
        if (adam->m[i] == NULL) {
            fprintf(stderr, "Error: Unable to allocate memory for Adam optimizer first moment vector[%d]\n", i);
            exit(1);
        }
        adam->v[i] = malloc(num_weights[i] * sizeof(float));
        if (adam->v[i] == NULL) {
            fprintf(stderr, "Error: Unable to allocate memory for Adam optimizer second moment vector[%d]\n", i);
            exit(1);
        }
        for (int j = 0; j < num_weights[i]; j++) {
            adam->m[i][j] = 0.0f;
            adam->v[i][j] = 0.0f;
        }
    }

    return adam;
}

RMSpropOptimizer* init_rmsprop(float learning_rate, float rho, float epsilon, int* num_weights, int num_layers) {
    RMSpropOptimizer* rmsprop = (RMSpropOptimizer*)malloc(sizeof(RMSpropOptimizer));
    if (rmsprop == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for RMSprop optimizer\n");
        exit(1);
    }

    rmsprop->learning_rate = learning_rate;
    rmsprop->rho = rho;
    rmsprop->epsilon = epsilon;

    rmsprop->square_avg_grad = malloc(num_layers * sizeof(float**));
    if (rmsprop->square_avg_grad == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for RMSprop optimizer squared average gradient\n");
        exit(1);
    }

    for (int i = 0; i < num_layers; i++) {
        rmsprop->square_avg_grad[i] = malloc(num_weights[i] * sizeof(float));
        if (rmsprop->square_avg_grad[i] == NULL) {
            fprintf(stderr, "Error: Unable to allocate memory for RMSprop optimizer squared average gradient[%d]\n", i);
            exit(1);
        }
        for (int j = 0; j < num_weights[i]; j++) {
            rmsprop->square_avg_grad[i][j] = 0.0f;
        }
    }

    return rmsprop;
}

void delete_optimizer(Optimizer* optimizer, int num_layers) {
    SGDOptimizer* sgd_optimizer;
    AdamOptimizer* adam_optimizer;
    RMSpropOptimizer* rmsprop_optimizer;

    switch (optimizer->type) {
        case SGD:
            sgd_optimizer = optimizer->optimizer.sgd;
            for (int i = 0; i < num_layers; i++) {
                free(sgd_optimizer->momentum_buffer[i]);
            }
            free(sgd_optimizer->momentum_buffer);
            free(sgd_optimizer);
            break;
        case ADAM:
            adam_optimizer = optimizer->optimizer.adam;
            for (int i = 0; i < num_layers; i++) {
                free(adam_optimizer->m[i]);
                free(adam_optimizer->v[i]);
            }
            free(adam_optimizer->m);
            free(adam_optimizer->v);
            free(adam_optimizer);
            break;
        case RMSPROP:
            rmsprop_optimizer = optimizer->optimizer.rmsprop;
            for (int i = 0; i < num_layers; i++) {
                free(rmsprop_optimizer->square_avg_grad[i]);
            }
            free(rmsprop_optimizer->square_avg_grad);
            free(rmsprop_optimizer);
            break;
        default:
            fprintf(stderr, "Error: Invalid optimizer type\n");
            exit(1);
    }
    free(optimizer);
}

