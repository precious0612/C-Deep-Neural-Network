//
//  layer.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/18/24.
//

#include "layer.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../../utils/utils.h"

Layer* create_layer(LayerType type, LayerParams params) {
    if (type < 0 || type >= LAYER_TYPE_COUNT) {
        fprintf(stderr, "Error: Invalid layer type\n");
        return NULL;
    }

    Layer* new_layer = (Layer*)malloc(sizeof(Layer));
    if (new_layer == NULL) {
        fprintf(stderr, "Error: Could not allocate memory for layer\n");
        return NULL;
    }

    new_layer->type       = type;
    new_layer->params     = params;
    new_layer->num_params = 0;
    new_layer->next_layer = NULL;
    new_layer->prev_layer = NULL;

    return new_layer;
}

static int is_valid_input_shape(Dimensions* input_shape) {
    if (input_shape == NULL || input_shape->width <= 0 || input_shape->height <= 0 || input_shape->channels <= 0) {
        return 0;
    } else {
        return 1;
    }
}

void initialize_layer(Layer* layer) {
    if (layer == NULL) {
        fprintf(stderr, "Error: Invalid layer pointer\n");
        return;
    }

    if (!is_valid_input_shape(&layer->input_shape)) {
        fprintf(stderr, "Error: Invalid input shape for layer\n");
        return;
    }

    switch (layer->type) {
        case CONVOLUTIONAL: {
            int num_filters    = layer->params.conv_params.num_filters;
            int filter_size    = layer->params.conv_params.filter_size;
            int input_channels = layer->input_shape.channels;

            if (num_filters <= 0 || filter_size <= 0 || input_channels <= 0) {
                fprintf(stderr, "Error: Invalid convolutional layer parameters\n");
                return;
            }

            layer->num_params = num_filters * input_channels * filter_size * filter_size + num_filters;

            layer->input = NULL;
            layer->input_before_activation = NULL;

            layer->weights.conv_weights = calloc_4d_float_array(num_filters, filter_size, filter_size, input_channels);
            if (layer->weights.conv_weights == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for convolutional weights\n");
                return;
            }

            layer->biases = (float*)calloc(num_filters, sizeof(float));
            if (layer->biases == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for convolutional biases\n");
                free_4d_float_array(layer->weights.conv_weights);
                return;
            }

            float scale = sqrt(2.0 / (float)(filter_size * filter_size * input_channels));
            float* weights_p = &layer->weights.conv_weights[0][0][0][0];
            for (int index = 0; index < num_filters * input_channels * filter_size * filter_size; ++index) {
                weights_p[index] = scale * rand_uniform(-1.0, 1.0);
            }

            layer->grads.conv_grads = calloc_4d_float_array(num_filters, filter_size, filter_size, input_channels);
            if (layer->grads.conv_grads == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for convolutional weight gradients\n");
                free_4d_float_array(layer->weights.conv_weights);
                free(layer->biases);
                return;
            }

            layer->bias_grads = (float*)calloc(num_filters, sizeof(float));
            if (layer->bias_grads == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for convolutional bias gradients\n");
                free_4d_float_array(layer->weights.conv_weights);
                free_4d_float_array(layer->grads.conv_grads);
                free(layer->biases);
                return;
            }
            break;
        }

        case POOLING:
            layer->input = NULL;
            break;

        case FULLY_CONNECTED: {
            int num_neurons = layer->params.fc_params.num_neurons;
            int input_size  = layer->input_shape.height * layer->input_shape.width * layer->input_shape.channels;

            if (num_neurons <= 0 || input_size <= 0) {
                fprintf(stderr, "Error: Invalid fully-connected layer parameters\n");
                return;
            }

            layer->num_params = num_neurons * input_size + num_neurons;

            layer->input = NULL;
            layer->input_before_activation = NULL;

            layer->weights.fc_weights = calloc_2d_float_array(num_neurons, input_size);
            if (layer->weights.fc_weights == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for fully-connected weights\n");
                return;
            }

            layer->biases = (float*)calloc(num_neurons, sizeof(float));
            if (layer->biases == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for fully-connected biases\n");
                free_2d_float_array(layer->weights.fc_weights);
                return;
            }

            float scale = sqrt(2.0 / (float)(input_size));
            for (int n = 0; n < num_neurons; n++) {
                for (int i = 0; i < input_size; i++) {
                    layer->weights.fc_weights[n][i] = scale * rand_uniform(-1.0, 1.0);
                }
            }

            layer->grads.fc_grads = calloc_2d_float_array(num_neurons, input_size);
            if (layer->grads.fc_grads == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for fully-connected weight gradients\n");
                free_2d_float_array(layer->weights.fc_weights);
                free(layer->biases);
                return;
            }

            layer->bias_grads = (float*)calloc(num_neurons, sizeof(float));
            if (layer->bias_grads == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for fully-connected bias gradients\n");
                free_2d_float_array(layer->weights.fc_weights);
                free_2d_float_array(layer->grads.fc_grads);
                free(layer->biases);
                return;
            }
            break;
        }

        case DROPOUT:
            layer->input = NULL;
            break;

        case ACTIVATION:
            layer->input = NULL;
            break;

        case FLATTEN:
            layer->input = NULL;
            break;

        default:
            fprintf(stderr, "Error: Unknown layer type\n");
            break;
    }
}

LayerOutput layer_forward_pass(Layer* layer, LayerInput input, int is_training) {

    if (layer == NULL) {
        fprintf(stderr, "Error: Invalid layer pointer\n");
        return NULL;
    }

    if (!is_valid_input_shape(&layer->input_shape)) {
        fprintf(stderr, "Error: Invalid input shape for layer\n");
        return NULL;
    }

    LayerOutput temp_output = NULL;
    LayerOutput output = NULL;
    switch (layer->type) {
        case CONVOLUTIONAL:
            if (is_training) {
                if (layer->input != NULL) {
                    free_3d_float_array(layer->input);
                }
                layer->input = copy_3d_float_array(input, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels);
            }
            temp_output = conv_forward(input, layer->weights.conv_weights, layer->biases,
                                       layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels,
                                       layer->params.conv_params.num_filters, layer->params.conv_params.filter_size, layer->params.conv_params.stride, layer->params.conv_params.padding);
            if (layer->params.conv_params.activation >= NONE) {
                output = temp_output;
                temp_output = NULL;
                break;
            }
            if (is_training) {
                if (layer->input_before_activation != NULL) {
                    free_3d_float_array(layer->input_before_activation);
                }
                layer->input_before_activation = copy_3d_float_array(temp_output, layer->output_shape.width, layer->output_shape.height, layer->output_shape.channels);
            }
            output = forward_activation(layer->params.conv_params.activation, temp_output, layer->output_shape.width, layer->output_shape.height, layer->output_shape.channels);
            free_3d_float_array(temp_output);
            temp_output = NULL;
            break;

        case POOLING:
            if (is_training) {
                if (layer->input != NULL) {
                    free_3d_float_array(layer->input);
                }
                layer->input = copy_3d_float_array(input, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels);
            }
            output = pool_forward(input, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels, layer->params.pooling_params.pool_size, layer->params.pooling_params.stride, layer->params.pooling_params.pool_type);
            break;

        case FULLY_CONNECTED:
            if (layer->input_shape.width > 1 || layer->input_shape.height > 1) {
                temp_output  = flatten(input, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels);
                if (is_training) {
                    if (layer->input != NULL) {
                        free_3d_float_array(layer->input);
                    }
                    layer->input = copy_3d_float_array(temp_output, 1, 1, layer->input_shape.width * layer->input_shape.height * layer->input_shape.channels);
                }
                output       = fc_forward(temp_output, layer->input_shape.width * layer->input_shape.height * layer->input_shape.channels, layer->params.fc_params.num_neurons, layer->weights.fc_weights, layer->biases);
                free_3d_float_array(temp_output);
                temp_output = output;
            } else {
                if (is_training) {
                    if (layer->input != NULL) {
                        free_3d_float_array(layer->input);
                    }
                    layer->input = copy_3d_float_array(input, 1, 1, layer->input_shape.channels);
                }
                temp_output = fc_forward(input, layer->input_shape.channels, layer->params.fc_params.num_neurons, layer->weights.fc_weights, layer->biases);
            }

            if (layer->params.fc_params.activation >= NONE) {
                output = temp_output;
                temp_output = NULL;
                break;
            }
            if (is_training) {
                if (layer->input_before_activation != NULL) {
                    free_3d_float_array(layer->input_before_activation);
                }
                layer->input_before_activation = copy_3d_float_array(temp_output, layer->output_shape.width, layer->output_shape.height, layer->output_shape.channels);
            }
            output = forward_activation(layer->params.fc_params.activation, temp_output, 1, 1, layer->output_shape.width * layer->output_shape.height * layer->output_shape.channels);
            free_3d_float_array(temp_output);
            temp_output = NULL;
            break;

        case DROPOUT:
            if (is_training) {
                if (layer->input != NULL) {
                    free_3d_float_array(layer->input);
                }
                layer->input = copy_3d_float_array(input, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels);
            }
            output = dropout_forward(input, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels, layer->params.dropout_params.dropout_rate);
            break;

        case ACTIVATION:
            if (is_training) {
                if (layer->input != NULL) {
                    free_3d_float_array(layer->input);
                }
                layer->input = copy_3d_float_array(input, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels);
            }
            output = forward_activation(layer->params.activation_params.activation, input, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels);
            break;

        case FLATTEN:
            output = flatten(input, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels);
            break;

        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            output = copy_3d_float_array(input, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels);
            break;
    }

    return output;
}

void layer_backward_pass(Layer* layer, LayerOutputGrad output_grad, LayerInputGrad input_grad) {

    if (layer == NULL) {
        fprintf(stderr, "Error: Invalid layer pointer\n");
        return;
    }

    if (output_grad == NULL || input_grad == NULL) {
        fprintf(stderr, "Error: Invalid input during layer backward\n");
        return;
    }

    switch (layer->type) {
        case CONVOLUTIONAL:
            if (layer->params.conv_params.activation <= NONE) {
                backward_activation(layer->params.conv_params.activation, layer->input_before_activation, output_grad, output_grad, layer->input_shape.width * layer->input_shape.height * layer->input_shape.channels);
            }
            conv_backward(layer->input, layer->weights.conv_weights, output_grad, input_grad, layer->grads.conv_grads, layer->bias_grads,
                          layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels,
                          layer->params.conv_params.num_filters, layer->params.conv_params.filter_size, layer->params.conv_params.stride, layer->params.conv_params.padding);
            break;

        case POOLING:
            pool_backward(layer->input, output_grad, input_grad,
                          layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels,
                          layer->params.pooling_params.pool_size, layer->params.pooling_params.stride, layer->params.pooling_params.pool_type);
            break;

        case FULLY_CONNECTED:
            if (layer->params.conv_params.activation <= NONE) {
                backward_activation(layer->params.fc_params.activation, layer->input_before_activation, output_grad, output_grad, layer->params.fc_params.num_neurons);
            }
            fc_backward(layer->input, output_grad, input_grad, layer->weights.fc_weights, layer->grads.fc_grads, layer->biases, layer->bias_grads,
                        layer->input_shape.width * layer->input_shape.height * layer->input_shape.channels, layer->params.fc_params.num_neurons);
            break;

        case DROPOUT:
            dropout_backward(layer->input, output_grad, input_grad, layer->input_shape.width * layer->input_shape.height * layer->input_shape.channels);
            break;

        case ACTIVATION:
            backward_activation(layer->params.activation_params.activation, layer->input, output_grad, input_grad, layer->input_shape.width * layer->input_shape.height * layer->input_shape.channels);
            break;

        case FLATTEN:
            free_3d_float_array(input_grad);
            input_grad = unflatten(output_grad, layer->input_shape.width, layer->input_shape.height, layer->input_shape.channels);
            break;

        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
}

void update_layer_weights(Layer* layer, Optimizer* optimizer, int layer_index) {

    if (layer == NULL || optimizer == NULL) {
        fprintf(stderr, "Error: Invalid layer pointer\n");
        return;
    }

    float*  weights_p;
    float*  weight_grads_p;

    int num_weights = 0;

    // MARK: SGD Pramas
    float* momentum_buffer_p;


    // MARK: Adam Params
    float* m_p;
    float* v_p;

    // MARK: RMSPROP
    float* square_avg_grad_p;

    switch (layer->type) {
        case CONVOLUTIONAL:
            switch (optimizer->type) {
                case SGD:
                    num_weights       = layer->params.conv_params.num_filters * layer->params.conv_params.filter_size * layer->params.conv_params.filter_size * layer->input_shape.channels;
                    weights_p         = &layer->weights.conv_weights[0][0][0][0];
                    weight_grads_p    = &layer->grads.conv_grads[0][0][0][0];
                    momentum_buffer_p = optimizer->optimizer.sgd->momentum_buffer[layer_index];
                    sgd(weights_p, weight_grads_p, optimizer->optimizer.sgd->momentum, momentum_buffer_p, optimizer->optimizer.sgd->learning_rate, num_weights);
                    sgd(layer->biases, layer->bias_grads, optimizer->optimizer.sgd->momentum, &momentum_buffer_p[num_weights], optimizer->optimizer.sgd->learning_rate, layer->params.conv_params.num_filters);
                    break;

                case ADAM:
                    num_weights    = layer->params.conv_params.num_filters * layer->params.conv_params.filter_size * layer->params.conv_params.filter_size * layer->input_shape.channels;
                    weights_p      = &layer->weights.conv_weights[0][0][0][0];
                    weight_grads_p = &layer->grads.conv_grads[0][0][0][0];
                    m_p            = optimizer->optimizer.adam->m[layer_index];
                    v_p            = optimizer->optimizer.adam->v[layer_index];
                    adam(weights_p, weight_grads_p, m_p, v_p,
                         optimizer->optimizer.adam->beta1, optimizer->optimizer.adam->beta2, optimizer->optimizer.adam->epsilon, optimizer->optimizer.adam->t, optimizer->optimizer.adam->learning_rate, num_weights);
                    adam(layer->biases, layer->bias_grads, &m_p[num_weights], &v_p[num_weights],
                         optimizer->optimizer.adam->beta1, optimizer->optimizer.adam->beta2, optimizer->optimizer.adam->epsilon, optimizer->optimizer.adam->t, optimizer->optimizer.adam->learning_rate, layer->params.conv_params.num_filters);
                    break;

                case RMSPROP:
                    num_weights       = layer->params.conv_params.num_filters * layer->params.conv_params.filter_size * layer->params.conv_params.filter_size * layer->input_shape.channels;
                    weights_p         = &layer->weights.conv_weights[0][0][0][0];
                    weight_grads_p    = &layer->grads.conv_grads[0][0][0][0];
                    square_avg_grad_p = optimizer->optimizer.rmsprop->square_avg_grad[layer_index];
                    rmsprop(weights_p, weight_grads_p, square_avg_grad_p,
                            optimizer->optimizer.rmsprop->rho, optimizer->optimizer.rmsprop->epsilon, optimizer->optimizer.rmsprop->learning_rate, num_weights);
                    rmsprop(layer->biases, layer->bias_grads, &square_avg_grad_p[num_weights],
                            optimizer->optimizer.rmsprop->rho, optimizer->optimizer.rmsprop->epsilon, optimizer->optimizer.rmsprop->learning_rate, layer->params.conv_params.num_filters);
                    break;

                default:
                    fprintf(stderr, "Undefined optimizer\n");
                    break;
            }
            break;
        case FULLY_CONNECTED:
            switch (optimizer->type) {
                case SGD:
                    num_weights       = layer->params.fc_params.num_neurons * layer->input_shape.height * layer->input_shape.width * layer->input_shape.channels;
                    weights_p         = &layer->weights.fc_weights[0][0];
                    weight_grads_p    = &layer->grads.fc_grads[0][0];
                    momentum_buffer_p = optimizer->optimizer.sgd->momentum_buffer[layer_index];
                    sgd(weights_p, weight_grads_p, optimizer->optimizer.sgd->momentum, momentum_buffer_p, optimizer->optimizer.sgd->learning_rate, num_weights);
                    sgd(layer->biases, layer->bias_grads, optimizer->optimizer.sgd->momentum, &momentum_buffer_p[num_weights], optimizer->optimizer.sgd->learning_rate, layer->params.fc_params.num_neurons);
                    break;

                case ADAM:
                    num_weights    = layer->params.fc_params.num_neurons * layer->input_shape.height * layer->input_shape.width * layer->input_shape.channels;
                    weights_p      = &layer->weights.fc_weights[0][0];
                    weight_grads_p = &layer->grads.fc_grads[0][0];
                    m_p            = optimizer->optimizer.adam->m[layer_index];
                    v_p            = optimizer->optimizer.adam->v[layer_index];
                    adam(weights_p, weight_grads_p, m_p, v_p,
                         optimizer->optimizer.adam->beta1, optimizer->optimizer.adam->beta2, optimizer->optimizer.adam->epsilon, optimizer->optimizer.adam->t, optimizer->optimizer.adam->learning_rate, num_weights);
                    adam(layer->biases, layer->bias_grads, &m_p[num_weights], &v_p[num_weights],
                         optimizer->optimizer.adam->beta1, optimizer->optimizer.adam->beta2, optimizer->optimizer.adam->epsilon, optimizer->optimizer.adam->t, optimizer->optimizer.adam->learning_rate, layer->params.fc_params.num_neurons);
                    break;

                case RMSPROP:
                    num_weights       = layer->params.fc_params.num_neurons * layer->input_shape.height * layer->input_shape.width * layer->input_shape.channels;
                    weights_p         = &layer->weights.fc_weights[0][0];
                    weight_grads_p    = &layer->grads.fc_grads[0][0];
                    square_avg_grad_p = optimizer->optimizer.rmsprop->square_avg_grad[layer_index];
                    rmsprop(weights_p, weight_grads_p, square_avg_grad_p,
                            optimizer->optimizer.rmsprop->rho, optimizer->optimizer.rmsprop->epsilon, optimizer->optimizer.rmsprop->learning_rate, num_weights);
                    rmsprop(layer->biases, layer->bias_grads, &square_avg_grad_p[num_weights],
                            optimizer->optimizer.rmsprop->rho, optimizer->optimizer.rmsprop->epsilon, optimizer->optimizer.rmsprop->learning_rate, layer->params.fc_params.num_neurons);
                    break;

                default:
                    fprintf(stderr, "Undefined optimizer\n");
                    break;
            }
            break;
        // Add cases for other layer types as needed
        default:
            break;
    }
}

void reset_layer_grads(Layer* layer) {

    float* grads_p;

    switch (layer->type) {
        case CONVOLUTIONAL:
            grads_p = &layer->grads.conv_grads[0][0][0][0];
            memset(grads_p, 0, layer->params.conv_params.num_filters * layer->params.conv_params.filter_size * layer->params.conv_params.filter_size * layer->input_shape.channels * sizeof(float));
            memset(layer->bias_grads, 0, layer->params.conv_params.num_filters * sizeof(float));
            break;
        case POOLING:
            break;
        case FULLY_CONNECTED:
            grads_p = &layer->grads.fc_grads[0][0];
            memset(grads_p, 0, layer->params.fc_params.num_neurons * layer->output_shape.channels * sizeof(float));
            memset(layer->bias_grads, 0, layer->params.fc_params.num_neurons * sizeof(float));
            break;
        case DROPOUT:
            break;
        case ACTIVATION:
            break;
        case FLATTEN:
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
}

void compute_output_shape(Layer* layer) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            {
                int filter_size = layer->params.conv_params.filter_size;
                int stride      = layer->params.conv_params.stride;
                int padding     = layer->params.conv_params.padding;
                int num_filters = layer->params.conv_params.num_filters;

                int input_height = layer->input_shape.height;
                int input_width  = layer->input_shape.width;

                int output_height = (input_height - filter_size + 2 * padding) / stride + 1;
                int output_width  = (input_width - filter_size + 2 * padding) / stride + 1;

                layer->output_shape.height   = output_height;
                layer->output_shape.width    = output_width;
                layer->output_shape.channels = num_filters;
            }
            break;
        case POOLING:
            {
                int pool_size = layer->params.pooling_params.pool_size;
                int stride    = layer->params.pooling_params.stride;

                int input_height   = layer->input_shape.height;
                int input_width    = layer->input_shape.width;
                int input_channels = layer->input_shape.channels;

                int output_height = (input_height - pool_size) / stride + 1;
                int output_width  = (input_width - pool_size) / stride + 1;

                layer->output_shape.height   = output_height;
                layer->output_shape.width    = output_width;
                layer->output_shape.channels = input_channels;
            }
            break;
        case FULLY_CONNECTED:
            {
                int num_neurons = layer->params.fc_params.num_neurons;

                layer->output_shape.height   = 1;
                layer->output_shape.width    = 1;
                layer->output_shape.channels = num_neurons;
            }
            break;
        case DROPOUT:
            {
                layer->output_shape = layer->input_shape;
            }
            break;
        case ACTIVATION:
            {
                layer->output_shape = layer->input_shape;
            }
            break;
        case FLATTEN:
            {
                layer->output_shape.height   = 1;
                layer->output_shape.width    = 1;
                layer->output_shape.channels = layer->input_shape.channels * layer->input_shape.height * layer->input_shape.width;
            }
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
}

void delete_layer(Layer* layer) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            free_4d_float_array(layer->weights.conv_weights);
            free_4d_float_array(layer->grads.conv_grads);
            free_3d_float_array(layer->input);
            free_3d_float_array(layer->input_before_activation);
            free(layer->biases);
            free(layer->bias_grads);
            break;
        case POOLING:
            free_3d_float_array(layer->input);
            break;
        case FULLY_CONNECTED:
            free_2d_float_array(layer->weights.fc_weights);
            free_2d_float_array(layer->grads.fc_grads);
            free_3d_float_array(layer->input);
            free_3d_float_array(layer->input_before_activation);
            free(layer->biases);
            free(layer->bias_grads);
            break;
        case DROPOUT:
            free_3d_float_array(layer->input);
            break;
        case ACTIVATION:
            free_3d_float_array(layer->input);
            break;
        case FLATTEN:
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type.\n");
            break;
    }
    free(layer);
}
