/* utils/optim.c */

void sgd(float weight, float grads, float momentum, float momentum_buffer, float learning_rate) {
    momentum_buffer = momentum * momentum_buffer + grads;
    weight -= learning_rate * momentum_buffer;
}

void adam(float weight, float grads, int t, float m, float v, float beta1, float beta2, float epsilon, float learning_rate) {
    m = beta1 * m + (1 - beta1) * grads;
    v = beta2 * v + (1 - beta2) * grads * grads;
    float m_hat = m / (1 - pow(beta1, t));
    float v_hat = v / (1 - pow(beta2, t));
    weight -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
}

void rmsprop(float weight, float grads, float square_avg_grad, float rho, float epsilon, float learning_rate) {
    square_avg_grad = rho * square_avg_grad + (1 - rho) * grads * grads;
    weight -= learning_rate * grads / (sqrt(square_avg_grad) + epsilon);
}
