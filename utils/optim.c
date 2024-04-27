/* utils/optim.c */

#include <math.h>

float sgd(float grads, float momentum, float momentum_buffer, float learning_rate) {
    momentum_buffer = momentum * momentum_buffer + grads;
    return learning_rate * momentum_buffer;
}

float adam(float grads, int t, float m, float v, float beta1, float beta2, float epsilon, float learning_rate) {
    m = beta1 * m + (1 - beta1) * grads;
    v = beta2 * v + (1 - beta2) * grads * grads;
    float m_hat = m / (1 - pow(beta1, t));
    float v_hat = v / (1 - pow(beta2, t));
    return learning_rate * m_hat / (sqrt(v_hat) + epsilon);
}

float rmsprop(float grads, float square_avg_grad, float rho, float epsilon, float learning_rate) {
    square_avg_grad = rho * square_avg_grad + (1 - rho) * grads * grads;
    return learning_rate * grads / (sqrt(square_avg_grad) + epsilon);
}
