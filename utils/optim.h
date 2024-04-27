/* utils/optim.h */

#ifndef OPTIM_H
#define OPTIM_H

float sgd(float grads, float momentum, float momentum_buffer, float learning_rate);
float adam(float grads, int t, float m, float v, float beta1, float beta2, float epsilon, float learning_rate);
float rmsprop(float grads, float square_avg_grad, float rho, float epsilon, float learning_rate);

#endif /* OPTIM_H */