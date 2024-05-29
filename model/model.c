//
//  model.c
//  Neural Network API
//
//  Created by 泽瑾瑜 on 5/24/24.
//

#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>

static void initialize_model(Model* model, Dimensions input, Dimensions output) {
    model->input      = input;
    model->output     = output;
    model->layers     = NULL;
    model->num_layers = 0;
    model->optimizer  = NULL;
    model->loss       = NULL;
}

Model* create_model(Dimensions input, Dimensions output) {
    Model* model = (Model*)malloc(sizeof(Model));
    if (model == NULL) {
        fprintf(stderr, "Error: Allocating memory failed for model\n");
        return NULL;
    }
    initialize_model(model, input, output);
    return model;
}

static void add_layer_to_model(Model* model, Layer* layer) {
    Layer** new_layers = (Layer**)realloc(model->layers, (model->num_layers + 1) * sizeof(Layer*));
    if (new_layers == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for layers.\n");
        return;
    }
    model->layers = new_layers;
    model->layers[model->num_layers] = layer;
    model->num_layers++;
}

void add_layer(Model* model, LayerType layer_type, int num_filters, int filter_size, int stride, int padding, ActivationType activation, PoolType pool_type, float dropout_rate) {
    Layer* layer;
    LayerParams params;
    
    switch (layer_type) {
        case CONVOLUTIONAL:
            params.conv_params.activation  = activation;
            params.conv_params.num_filters = num_filters;
            params.conv_params.filter_size = filter_size;
            params.conv_params.stride      = stride;
            params.conv_params.padding     = padding;
            
            layer = create_layer(CONVOLUTIONAL, params);
            add_layer_to_model(model, layer);
            break;
            
        case POOLING:
            params.pooling_params.pool_type = pool_type;
            params.pooling_params.pool_size = filter_size;
            params.pooling_params.stride    = stride;
            
            layer = create_layer(POOLING, params);
            add_layer_to_model(model, layer);
            break;
            
        case FULLY_CONNECTED:
            params.fc_params.num_neurons = num_filters;
            params.fc_params.activation  = activation;
            
            layer = create_layer(FULLY_CONNECTED, params);
            add_layer_to_model(model, layer);
            break;
            
        case DROPOUT:
            params.dropout_params.dropout_rate = dropout_rate;

            layer = create_layer(DROPOUT, params);
            add_layer_to_model(model, layer);
            break;
            
        case ACTIVATION:
            params.activation_params.activation = activation;
            
            layer = create_layer(ACTIVATION, params);
            add_layer_to_model(model, layer);
            break;
            
        case FLATTEN:
            layer = create_layer(FLATTEN, (LayerParams){});
            add_layer_to_model(model, layer);
            break;
            
        default:
            fprintf(stderr, "Error: Invalid layer type specified.\n");
            break;
    }
}

void compile_model(Model* model, OptimizerType optimizer_type, LearningRate learning_rate, LossType loss_type, Metric metric) {
    // Check if the model pointer is valid
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    // Assign the configuration settings to the model
    model->learning_rate = learning_rate;
    model->loss          = init_loss_function(loss_type);
    model->metric        = metric;

    int num_layers = model->num_layers;
    int* num_weights = (int *)calloc(num_layers, sizeof(int));
    if (num_weights == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for num_weights array\n");
        delete_loss_function(model->loss);
        return;
    }

    // Set the input shape of the first layer
    if (model->num_layers > 0) {
        Layer* first_layer = model->layers[0];
        if (first_layer == NULL) {
            fprintf(stderr, "Error: First layer pointer is NULL.\n");
            delete_optimizer(model->optimizer, num_layers);
            delete_loss_function(model->loss);
            return;
        }
        first_layer->input_shape.width    = model->input.width;
        first_layer->input_shape.height   = model->input.height;
        first_layer->input_shape.channels = model->input.channels;
    } else {
        fprintf(stderr, "Error: Model does not have any layers.\n");
        free(num_weights);
        num_weights = NULL;
        delete_optimizer(model->optimizer, num_layers);
        delete_loss_function(model->loss);
        return;
    }

    Layer* prev_layer = NULL;
    // Compute the output shapes for each layer
    for (int i = 0; i < model->num_layers; i++) {
        Layer* current_layer = model->layers[i];
        if (current_layer == NULL) {
            fprintf(stderr, "Error: Layer pointer is NULL.\n");
            free(num_weights);
            num_weights = NULL;
            delete_optimizer(model->optimizer, num_layers);
            delete_loss_function(model->loss);
            for (int j = 0; j <= i; ++j) {
                delete_layer(model->layers[j]);
            }
            free(model->layers);
            model->layers = NULL;
            return;
        }
        current_layer->prev_layer = prev_layer;
        prev_layer = current_layer;
        current_layer->next_layer = (i < model->num_layers - 1) ? model->layers[i + 1] : NULL;
        initialize_layer(current_layer);
        compute_output_shape(current_layer);
        if (i < model->num_layers - 1) {
            Layer* next_layer = model->layers[i + 1];
            if (next_layer == NULL) {
                fprintf(stderr, "Error: Next layer pointer is NULL.\n");
                free(num_weights);
                num_weights = NULL;
                delete_optimizer(model->optimizer, num_layers);
                delete_loss_function(model->loss);
                for (int j = 0; j <= i; ++j) {
                    delete_layer(model->layers[j]);
                }
                free(model->layers);
                model->layers = NULL;
                return;
            }
            next_layer->input_shape = current_layer->output_shape;
        }
    }
    
    for (int i = 0; i < num_layers; ++i) {
        Layer* current_layer = model->layers[i];
        if (current_layer == NULL) {
            fprintf(stderr, "Error: Layer pointer is NULL.\n");
            free(num_weights);
            num_weights = NULL;
            delete_loss_function(model->loss);
            return;
        }

        num_weights[i] = current_layer->num_params;
    }

    // Create and initialize the optimizer
    model->optimizer = create_optimizer(optimizer_type, learning_rate, num_weights, num_layers);
    if (model->optimizer == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for optimizer\n");
        free(num_weights);
        num_weights = NULL;
        delete_loss_function(model->loss);
        return;
    }

    free(num_weights);
    num_weights = NULL;

    // Check if the final layer output shape matches the output information
    Layer* final_layer = model->layers[model->num_layers - 1];
    if (final_layer == NULL) {
        fprintf(stderr, "Error: Final layer pointer is NULL.\n");
        delete_optimizer(model->optimizer, num_layers);
        delete_loss_function(model->loss);
        for (int i = 0; i <= model->num_layers; ++i) {
            delete_layer(model->layers[i]);
        }
        free(model->layers);
        model->layers = NULL;
        return;
    }
    if (final_layer->output_shape.width    == model->output.width &&
        final_layer->output_shape.height   == model->output.height &&
        final_layer->output_shape.channels == model->output.channels) {
        // Output shape matches
        printf("*********************************************\n");
        printf("\nModel compiled successfully!\n");
        print_model_info(model);
        printf("\n*********************************************\n");
    } else {
        // Output shape does not match
        fprintf(stderr, "Error: Final layer output shape does not match the output information.\n");
        delete_optimizer(model->optimizer, num_layers);
        delete_loss_function(model->loss);
        for (int i = 0; i <= model->num_layers; ++i) {
            delete_layer(model->layers[i]);
        }
        free(model->layers);
        model->layers = NULL;
    }
}

static void print_model_config(const Model* model) {
    printf("Model Configuration:\n");
    printf("Input Shape: (%d, %d, %d)\n", model->input.width, model->input.height, model->input.channels);
    switch (model->optimizer->type) {
        case SGD:
            printf("Optimizer: SGD(Stochastic gradient descent)\n");
            break;
        
        case ADAM:
            printf("Optimizer: Adam(Adaptive Moment Estimation)\n");
            break;
            
        case RMSPROP:
            printf("Optimizer: RMSProp(Root Mean Square Propagation)\n");
            break;
            
        default:
            printf("Optimizer: Undefined optimizer\n");
            break;
    }
    printf("Learning Rate: %.6f\n", model->learning_rate);
    switch (model->loss->type) {
        case CrossEntropy:
            printf("Loss Function: cross-entropy loss\n");
            break;
            
        case MSE:
            printf("Loss Function: MSE(mean squared error)\n");
            break;
            
        default:
            printf("Loss Function: Undefined loss function\n");
            break;
    }
    switch (model->metric) {
        case ACCURACY:
            printf("Evaluation Metric: Accuracy\n");
            break;
            
        case LOSS:
            printf("Evaluation Metric: Loss\n");
            break;
            
        default:
            printf("Evaluation Metric: Undefined metric\n");
            break;
    }
}

static void print_layer_activation(ActivationType activation) {
    switch (activation) {
        case RELU:
            printf("  Activation Function: ReLU(rectified linear unit)\n");
            break;
            
        case SIGMOID:
            printf("  Activation Function: Sigmoid\n");
            break;
            
        case TANH:
            printf("  Activation Function: Tanh(Hyperbolic tangent)\n");
            break;
            
        case SOFTMAX:
            printf("  Activation Function: Softmax\n");
            break;
            
        default:
            printf("  Activation Function: Undefined activation function\n");
            break;
    }
}

static void print_layer_info(const Layer* layer, int layer_num) {
    if (layer == NULL) {
        fprintf(stderr, "Error: Layer pointer is NULL.\n");
        return;
    }

    printf("\nLayer %d: ", layer_num);

    switch (layer->type) {
        case CONVOLUTIONAL:
            printf("Convolutional\n");
            printf("  Parameter number: %d\n", layer->num_params);
            printf("  Output Shape: (%d, %d, %d)\n", layer->output_shape.width, layer->output_shape.height, layer->output_shape.channels);
            printf("  Number of Filters: %d\n", layer->params.conv_params.num_filters);
            printf("  Filter Size: %d\n", layer->params.conv_params.filter_size);
            printf("  Stride: %d\n", layer->params.conv_params.stride);
            printf("  Padding: %d\n", layer->params.conv_params.padding);
            print_layer_activation(layer->params.conv_params.activation);
            break;
        case POOLING:
            printf("Pooling\n");
            printf("  Output Shape: (%d, %d, %d)\n", layer->output_shape.width, layer->output_shape.height, layer->output_shape.channels);
            printf("  Pool Size: %d\n", layer->params.pooling_params.pool_size);
            printf("  Stride: %d\n", layer->params.pooling_params.stride);
            switch (layer->params.pooling_params.pool_type) {
                case MAX:
                    printf("  Pool Type: max\n");
                    break;
                    
                case AVARAGE:
                    printf("  Pool Type: avarage\n");
                    break;
                    
                default:
                    printf("  Pool Type: Undefined pool type\n");
                    break;
            }
            break;
        case FULLY_CONNECTED:
            printf("Fully Connected\n");
            printf("  Parameter number: %d\n", layer->num_params);
            printf("  Output Shape: (%d, %d, %d)\n", layer->output_shape.width, layer->output_shape.height, layer->output_shape.channels);
            printf("  Number of Neurons: %d\n", layer->params.fc_params.num_neurons);
            print_layer_activation(layer->params.fc_params.activation);
            break;
        case DROPOUT:
            printf("Dropout\n");
            printf("  Output Shape: (%d, %d, %d)\n", layer->output_shape.width, layer->output_shape.height, layer->output_shape.channels);
            printf("  Dropout Rate: %.2f\n", layer->params.dropout_params.dropout_rate);
            break;
        case ACTIVATION:
            printf("Activation\n");
            print_layer_activation(layer->params.activation_params.activation);
            break;
        case FLATTEN:
            printf("Flatten\n");
            printf("  Output Shape: (%d, %d, %d)\n", layer->output_shape.width, layer->output_shape.height, layer->output_shape.channels);
            break;
        default:
            printf("Unknown Layer Type\n");
            break;
    }
}

void print_model_info(const Model* model) {
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    print_model_config(model);

    int layer_num = 1;
    if (model->num_layers > 0) {
        Layer* first_layer = model->layers[0];
        if (first_layer == NULL) {
            fprintf(stderr, "Error: First layer pointer is NULL.\n");
            return;
        }

        printf("\nInput Shape: (%d, %d, %d)\n", first_layer->input_shape.width, first_layer->input_shape.height, first_layer->input_shape.channels);
        print_layer_info(first_layer, layer_num);
        layer_num++;
    }

    for (int i = 1; i < model->num_layers; ++i) {
        Layer* current_layer = model->layers[i];
        if (current_layer == NULL) {
            fprintf(stderr, "Error: Layer %d pointer is NULL.\n", i);
            return;
        }
        print_layer_info(current_layer, layer_num);
        layer_num++;
    }
}

Output forward_pass(Model* model, Input input) {
    
    Input temp_input = copy_3d_float_array(input, model->input.width, model->input.height, model->input.channels);
    Output output    = NULL;
    
    for (int i = 0; i < model->num_layers; ++i) {
        output = layer_forward_pass(model->layers[i], temp_input, 1);
        free_3d_float_array(temp_input);
        temp_input = output;
    }
    
    return output;
}

static Output forward_pass_for_evaluate(Model* model, Input input) {
    
    Input temp_input = copy_3d_float_array(input, model->input.width, model->input.height, model->input.channels);
    Output output    = NULL;
    
    for (int i = 0; i < model->num_layers; ++i) {
        output = layer_forward_pass(model->layers[i], temp_input, 0);
        free_3d_float_array(temp_input);
        temp_input = output;
    }
    
    return output;
}

static void backward_pass(Model* model, OutputGrad output_grad) {
    
    OutputGrad temp_output_grad = copy_3d_float_array(output_grad, model->output.width, model->output.height, model->output.channels);
    
    for (int i = model->num_layers - 1; i >= 0; --i) {
        InputGrad input_grad = calloc_3d_float_array(model->layers[i]->input_shape.width, model->layers[i]->input_shape.height, model->layers[i]->input_shape.channels);
        layer_backward_pass(model->layers[i], temp_output_grad, input_grad);
        free_3d_float_array(temp_output_grad);
        temp_output_grad = input_grad;
        input_grad = NULL;
    }
    
    free_3d_float_array(temp_output_grad);
}

static BatchedOutputs forward_pass_by_batches(Model* model, BatchedInputs batched_inputs, BatchSize batch_size) {
    
    BatchedOutputs batched_outputs = (float****)malloc(sizeof(float***) * batch_size);
    for (int i = 0; i < batch_size; ++i) {
        batched_outputs[i] = forward_pass(model, batched_inputs[i]);
    }
    return batched_outputs;
}

static void backward_pass_by_batches(Model* model, BatchedOutputGrads batched_output_grads, BatchSize batch_size) {
    
    for (int i = 0; i < batch_size; ++i) {
        backward_pass(model, batched_output_grads[i]);
    }
}

static void update_model_weights(Model* model) {
    
    for (int i = 0; i < model->num_layers; ++i) {
        update_layer_weights(model->layers[i], model->optimizer, i);
    }
}

void reset_model_grads(Model* model) {
    Layer* layer = model->layers[0];
    while (layer != NULL) {
        reset_layer_grads(layer);
        layer = layer->next_layer;
    }
}

static void free_batch_resources(Model* model, BatchedOutputs batched_outputs, BatchedOutputGrads batched_output_grads, BatchSize batch_size) {
    
    if (batched_outputs != NULL) {
        for (int i = 0; i < batch_size; i++) {
            free_3d_float_array(batched_outputs[i]);
        }
        free(batched_outputs);
        batched_outputs = NULL;
    }

    if (batched_output_grads != NULL) {
        for (int i = 0; i < batch_size; i++) {
            free_3d_float_array(batched_output_grads[i]);
        }
        free(batched_output_grads);
        batched_output_grads = NULL;
    }
}

static LossValue compute_loss(Output output, Label label, int num_classes, LossType type) {
    
    switch (type) {
        case CrossEntropy:
            return categorical_crossentropy_loss(&output[0][0][0], label, num_classes);
            
        case MSE:
            return mean_squared_error_loss(&output[0][0][0], label, num_classes);
            
        default:
            fprintf(stderr, "Error: Undefined Loss Type");
            return 0.0f;
    }
}

static LossValue compute_loss_by_batches(BatchedOutputs batched_outputs, Label* batched_labels, LossType type, BatchSize batch_size, int num_classes) {
    
    LossValue total_loss = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        compute_loss(batched_outputs[i], batched_labels[i], num_classes, type);
    }
    
    return total_loss / batch_size;
}

static void compute_output_grad(OutputGrad output_grad, Output output, Label label, int num_classes, Dimensions output_shape) {
    
    float* output_p        = &output[0][0][0];
    float* output_grad_p   = &output_grad[0][0][0];
    
    for (int i = 0; i < num_classes; ++i) {
        if (i == label) {
            output_grad_p[i] = output_p[i] - 1.0f;
        } else {
            output_grad_p[i] = output_p[i];
        }
    }
}

static void compute_output_grad_by_batches(BatchedOutputGrads batched_output_grads, BatchedOutputs batched_outputs, Label* batched_labels, int num_classes, BatchSize batch_size, Dimensions output_shape) {
    
    for (int i = 0; i < batch_size; ++i) {
        compute_output_grad(batched_output_grads[i], batched_outputs[i], batched_labels[i], num_classes, output_shape);
    }
}

static Label get_prediction_as_metric(Output output, const Metric metric, int num_classes) {
    
    float* output_p = &output[0][0][0];
    
    switch (metric) {
        case ACCURACY:
            return get_prediction(output_p, num_classes);
            
        case LOSS:
            return get_prediction(output_p, num_classes);
            
        default:
            return -1;
    }
}

static float compute_accuracy_by_batches(BatchedOutputs batched_outputs, Label* batched_labels, BatchSize batch_size, int num_classes) {
    
    int correct_predictions = 0;
    
    for (int i = 0; i < batch_size; ++i) {
        if (get_prediction_as_metric(batched_outputs[i], ACCURACY, num_classes) == batched_labels[i]) {
            correct_predictions++;
        }
    }
    
    return ((float)correct_predictions) / ((float)batch_size);
}

void train_model(Model* model, Dataset* dataset, int num_epochs) {
    
    if (model == NULL || dataset == NULL) {
        fprintf(stderr, "Error: Model or dataset pointer is NULL.\n");
        return;
    }
    
    if (model->input.width != dataset->data_dimensions.width || model->input.height != dataset->data_dimensions.height || model->input.channels != dataset->data_dimensions.channels) {
        fprintf(stderr, "Error: Model and dataset dimensions are not matched!\n");
        return;
    }

    int                num_classes          = model->output.width * model->output.height * model->output.channels;
    BatchedInputs      batched_inputs       = NULL;
    BatchedOutputs     batched_outputs      = NULL;
    BatchedOutputGrads batched_output_grads = (float****)malloc(dataset->batch_size * sizeof(float***));
    if (batched_output_grads == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for batch grads.\n");
        return;
    }
    
    for (int i = 0; i < dataset->batch_size; ++i) {
        batched_output_grads[i] = calloc_3d_float_array(model->output.width, model->output.height, model->output.channels);
        if (batched_output_grads[i] == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for %dth batch grad.\n", i);
            free(batched_output_grads);
            batched_output_grads = NULL;
            return;
        }
    }
    
    int total_batches = 0;
    Dataset* temp_dataset = dataset;

    // Calculate the total number of batches
    while (temp_dataset != NULL) {
        total_batches++;
        temp_dataset = temp_dataset->next_batch;
    }
    
    temp_dataset = NULL;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Dataset* batch    = dataset;
        int batch_count   = 0;

        int print_interval = PRINT_INTERVAL(total_batches);
        
        while (batch != NULL) {
            int batch_num = batch_count++;
            batched_inputs = (float****)malloc(sizeof(float***) * batch->batch_size);
            if (batched_inputs == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for batch inputs.\n");
                free_batch_resources(model, batched_outputs, batched_output_grads, batch->batch_size);
                return;
            }

            for (int i = 0; i < batch->batch_size; i++) {
                if (i >= batch->num_images) {
                    fprintf(stderr, "Error: Batch size exceeds the number of images in the batch.\n");
                    free_batch_resources(model, batched_outputs, batched_output_grads, batch->batch_size);
                    free(batched_inputs);
                    batched_inputs = NULL;
                    return;
                }

                switch (batch->data_type) {
                    case INT:
                        batched_inputs[i] = copy_3d_float_array_from_int(batch->images[i]->int_data, model->input.width, model->input.height, model->input.channels);
                        break;
                    case FLOAT32:
                        batched_inputs[i] = batch->images[i]->float32_data;
                        break;
                    default:
                        fprintf(stderr, "Error: Invalid data type specified.\n");
                        free_batch_resources(model, batched_outputs, batched_output_grads, batch->batch_size);
                        free(batched_inputs);
                        batched_inputs = NULL;
                        return;
                }
            }

            batched_outputs = forward_pass_by_batches(model, batched_inputs, batch->batch_size);
            if (batched_outputs == NULL) {
                fprintf(stderr, "Error: Forward pass failed.\n");
                free_batch_resources(model, batched_outputs, batched_output_grads, batch->batch_size);
                free(batched_inputs);
                batched_inputs = NULL;
                return;
            }

            Label* batched_labels = batch->labels;
            compute_output_grad_by_batches(batched_output_grads, batched_outputs, batched_labels, num_classes, batch->batch_size, model->output);
            backward_pass_by_batches(model, batched_output_grads, batch->batch_size);
            update_model_weights(model);

            if (total_batches <= MAX_BATCH_PRINT || (batch_num + 1) % print_interval == 0 || batch_num == total_batches - 1) {
                // Print progress
                printf("\nEpoch %d, Batch %d/%d, DataNum %d\n", epoch + 1, batch_num + 1, total_batches, batch->num_images);

                // Print loss and accuracy for the batch
                LossValue loss    = compute_loss_by_batches(batched_outputs, batched_labels, model->loss->type, batch->batch_size, num_classes);
                Accuracy accuracy = compute_accuracy_by_batches(batched_outputs, batched_labels, batch->batch_size, num_classes);
                printf("Epoch %d, Batch Loss: %.6f, Batch Accuracy: %.2f%%\n", epoch + 1, loss, accuracy * 100.0f);
            }

            for (int i = 0; i < batch->batch_size; ++i) {
                free_3d_float_array(batched_outputs[i]);
            }
            free(batched_outputs);
            batched_outputs = NULL;
            free(batched_inputs);
            batched_inputs = NULL;

            // Print progress bar for the current epoch
            float epoch_progress = ((float)batch_count) / ((float)total_batches);
            print_progress_bar(epoch_progress, 100);

            batch = batch->next_batch;
        }

        // Print progress
        printf("\nEpoch %d completed.\n", epoch + 1);

        // Evaluate the model on the validation dataset
        if (dataset->val_dataset != NULL) {
            Accuracy val_accuracy = evaluate_model(model, dataset->val_dataset);
            printf("\nValidation Accuracy: %.2f%%\n", val_accuracy * 100.0f);
        }
    }
    
    if (batched_output_grads != NULL) {
        for (int i = 0; i < dataset->batch_size; ++i) {
            free_3d_float_array(batched_output_grads[i]);
        }
        free(batched_output_grads);
        batched_output_grads = NULL;
    }
}

Accuracy evaluate_model(Model* model, Dataset* dataset) {
    if (model == NULL || dataset == NULL) {
        fprintf(stderr, "Error: Model or dataset pointer is NULL.\n");
        return 0.0f;
    }

    int correct_predictions = 0;
    int total_samples       = 0;
    int batch_count         = 0;
    int total_batches       = 0;
    int num_classes         = model->output.width * model->output.height * model->output.channels;
    
    Dataset* temp_dataset = dataset;

    // Calculate the total number of batches
    while (temp_dataset != NULL) {
        total_batches++;
        temp_dataset = temp_dataset->next_batch;
    }
    
    temp_dataset = NULL;
    
    Dataset* batch = dataset;

    Output output = NULL;

    while (batch != NULL) {
        char batch_progress_str[MAX_BATCH_PROGRESS_STR_LEN];
        snprintf(batch_progress_str, MAX_BATCH_PROGRESS_STR_LEN, "\nEvaluating Batch %d/%d", batch_count + 1, total_batches);
        printf("%s\n", batch_progress_str);

        for (int i = 0; i < batch->num_images; i++) {
            if (i >= batch->num_images) {
                char error_msg[MAX_ERROR_MSG_LEN];
                snprintf(error_msg, MAX_ERROR_MSG_LEN, "Error: Batch size exceeds the number of images in batch %d.\n", batch_count + 1);
                fprintf(stderr, "%s", error_msg);
                if (output != NULL) {
                    free_3d_float_array(output);
                }
                return 0.0f;
            }

            switch (batch->data_type) {
                case INT:
                    output = forward_pass_for_evaluate(model, copy_3d_float_array_from_int(batch->images[i]->int_data, model->input.width, model->input.height, model->input.channels));
                    if (output == NULL) {
                        fprintf(stderr, "Error: Forward pass failed for image %d in batch %d.\n", i + 1, batch_count + 1);
                        return 0.0f;
                    }
                    break;
                case FLOAT32:
                    output = forward_pass_for_evaluate(model, batch->images[i]->float32_data);
                    if (output == NULL) {
                        fprintf(stderr, "Error: Forward pass failed for image %d in batch %d.\n", i + 1, batch_count + 1);
                        return 0.0f;
                    }
                    break;
                default:
                    fprintf(stderr, "Error: Invalid data type specified.\n");
                    return 0.0f;
            }

            int prediction = get_prediction_as_metric(output, ACCURACY, num_classes);
            if (prediction == batch->labels[i]) {
                correct_predictions++;
            }
            total_samples++;
            
            free_3d_float_array(output);
        }

        // Print progress bar for the current batch
        float batch_progress = ((float)(batch_count + 1)) / ((float)total_batches);
        print_progress_bar(batch_progress, 100);

        batch_count++;
        batch = batch->next_batch;
    }

    printf("\n");
    return ((float)correct_predictions) / ((float)total_samples);
}

void delete_model(Model* model) {
    // Check if the model pointer is valid
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    // Check if the layers array is valid
    if (model->layers != NULL) {
        // Delete each layer and free its memory
        for (int i = 0; i < model->num_layers; i++) {
            if (model->layers[i] == NULL) {
                break;
            }
            delete_layer(model->layers[i]);
        }
        
        // Free the memory allocated for the layers array
        free(model->layers);
        model->layers = NULL;
    }

    // Delete the optimizer and free its memory
    if (model->optimizer != NULL) {
        delete_optimizer(model->optimizer, model->num_layers);
    }
    
    if (model->loss != NULL) {
        delete_loss_function(model->loss);
    }

    // Free the memory allocated for the model struct
    free(model);
    model = NULL;
}

static const char* get_layer_type_string(LayerType type) {
    switch (type) {
        case CONVOLUTIONAL:
            return "conv";
        case POOLING:
            return "pool";
        case FULLY_CONNECTED:
            return "fc";
        case DROPOUT:
            return "dropout";
        case ACTIVATION:
            return "activation";
        case FLATTEN:
            return "flatten";
        // Add other layer types as needed
        default:
            return "unknown";
    }
}

static herr_t save_conv_weights(hid_t group_id, float ****weights, int num_filters, int filter_size, int channels) {
    if (weights == NULL || num_filters <= 0 || filter_size <= 0 || channels <= 0) {
        fprintf(stderr, "Error: Invalid input parameters for save_conv_weights.\n");
        return -1;
    }

    hid_t dataset_id, dataspace_id;
    hsize_t dims[4] = {num_filters, channels, filter_size, filter_size};
    char dataset_name[MAX_DATASET_NAME_LEN];

    dataspace_id = H5Screate_simple(4, dims, NULL);
    if (dataspace_id < 0) {
        fprintf(stderr, "Error creating HDF5 dataspace.\n");
        return -1;
    }

    snprintf(dataset_name, MAX_DATASET_NAME_LEN, "conv_weights");
    dataset_id = H5Dcreate(group_id, dataset_name, H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error creating HDF5 dataset: %s\n", dataset_name);
        H5Sclose(dataspace_id);
        return -1;
    }
    
    float* weights_p = &weights[0][0][0][0];

    herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, weights_p);
    if (status < 0) {
        fprintf(stderr, "Error writing HDF5 dataset: %s\n", dataset_name);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        return -1;
    }

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    return 0;
}

static herr_t save_fc_weights(hid_t group_id, float **weights, int num_neurons, int input_size) {
    if (weights == NULL || num_neurons <= 0 || input_size <= 0) {
        fprintf(stderr, "Error: Invalid input parameters for save_fc_weights.\n");
        return -1;
    }

    hid_t dataset_id, dataspace_id;
    hsize_t dims[2] = {num_neurons, input_size};
    char dataset_name[MAX_DATASET_NAME_LEN];

    dataspace_id = H5Screate_simple(2, dims, NULL);
    if (dataspace_id < 0) {
        fprintf(stderr, "Error creating HDF5 dataspace.\n");
        return -1;
    }

    snprintf(dataset_name, MAX_DATASET_NAME_LEN, "fc_weights");
    dataset_id = H5Dcreate(group_id, dataset_name, H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error creating HDF5 dataset: %s\n", dataset_name);
        H5Sclose(dataspace_id);
        return -1;
    }
    
    float* weights_p = &weights[0][0];

    herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, weights_p);
    if (status < 0) {
        fprintf(stderr, "Error writing HDF5 dataset: %s\n", dataset_name);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        return -1;
    }

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    return 0;
}

static herr_t save_biases(hid_t group_id, float *biases, int num_biases) {
    if (biases == NULL || num_biases <= 0) {
        fprintf(stderr, "Error: Invalid input parameters for save_biases.\n");
        return -1;
    }

    hid_t dataset_id, dataspace_id;
    hsize_t dims[1] = {num_biases};
    char dataset_name[MAX_DATASET_NAME_LEN];

    dataspace_id = H5Screate_simple(1, dims, NULL);
    if (dataspace_id < 0) {
        fprintf(stderr, "Error creating HDF5 dataspace.\n");
        return -1;
    }

    snprintf(dataset_name, MAX_DATASET_NAME_LEN, "biases");
    dataset_id = H5Dcreate(group_id, dataset_name, H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error creating HDF5 dataset: %s\n", dataset_name);
        H5Sclose(dataspace_id);
        return -1;
    }

    herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, biases);
    if (status < 0) {
        fprintf(stderr, "Error writing HDF5 dataset: %s\n", dataset_name);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        return -1;
    }

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    return 0;
}

static herr_t save_layer_weights(hid_t group_id, Layer* layer) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            // Save convolutional layer weights and biases
            if (save_conv_weights(group_id, layer->weights.conv_weights, layer->params.conv_params.num_filters, layer->params.conv_params.filter_size, layer->input_shape.channels) != 0) {
                return -1;
            }
            if (save_biases(group_id, layer->biases, layer->params.conv_params.num_filters) != 0) {
                return -1;
            }
            break;
        case POOLING:
            // Pooling layer does not have any weights or biases
            break;
        case FULLY_CONNECTED:
            // Save fully connected layer weights and biases
            if (save_fc_weights(group_id, layer->weights.fc_weights, layer->params.fc_params.num_neurons, layer->input_shape.width * layer->input_shape.height * layer->input_shape.channels) != 0) {
                return -1;
            }
            if (save_biases(group_id, layer->biases, layer->params.fc_params.num_neurons) != 0) {
                return -1;
            }
            break;
        // Add other layer types as needed
        default:
            fprintf(stderr, "Error: Unsupported layer type.\n");
            return -1;
    }

    return 0;
}

void save_model_weights(Model* model, const char* filename) {
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error creating HDF5 file: %s\n", filename);
        return;
    }

    for (int i = 0; i < model->num_layers; i++) {
        Layer* layer = model->layers[i];
        if (layer == NULL) {
            fprintf(stderr, "Error: Layer %d pointer is NULL.\n", i);
            H5Fclose(file_id);
            return;
        }

        char group_name[MAX_GROUP_NAME_LEN];
        snprintf(group_name, MAX_GROUP_NAME_LEN, "/layer_%d_%s", i, get_layer_type_string(layer->type));

        hid_t group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (group_id < 0) {
            fprintf(stderr, "Error creating HDF5 group: %s\n", group_name);
            H5Fclose(file_id);
            return;
        }

        herr_t save_status = save_layer_weights(group_id, layer);
        if (save_status != 0) {
            fprintf(stderr, "Error saving weights for layer %d\n", i);
            H5Gclose(group_id);
            H5Fclose(file_id);
            return;
        }

        H5Gclose(group_id);
    }

    H5Fclose(file_id);
}

static herr_t load_conv_weights(hid_t group_id, float *****weights, int num_filters, int filter_size, int channels) {
    if (weights == NULL || num_filters <= 0 || filter_size <= 0 || channels <= 0) {
        fprintf(stderr, "Error: Invalid input parameters for load_conv_weights.\n");
        return -1;
    }

    hid_t dataset_id, dataspace_id;
    char dataset_name[MAX_DATASET_NAME_LEN];

    snprintf(dataset_name, MAX_DATASET_NAME_LEN, "conv_weights");
    dataset_id = H5Dopen(group_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening HDF5 dataset: %s\n", dataset_name);
        return -1;
    }

    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        fprintf(stderr, "Error getting HDF5 dataspace.\n");
        H5Dclose(dataset_id);
        return -1;
    }

    if (*weights == NULL) {
        *weights = (float ****) malloc(sizeof(float ***) * num_filters);
        if (*weights == NULL) {
            fprintf(stderr, "Error allocating memory for conv weights.\n");
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
            return -1;
        }

        for (int i = 0; i < num_filters; i++) {
            (*weights)[i] = (float ***) malloc(sizeof(float **) * channels);
            if ((*weights)[i] == NULL) {
                fprintf(stderr, "Error allocating memory for conv weights.\n");
                for (int j = 0; j < i; j++) {
                    free((*weights)[j]);
                }
                free(*weights);
                H5Dclose(dataset_id);
                H5Sclose(dataspace_id);
                return -1;
            }

            for (int j = 0; j < channels; j++) {
                (*weights)[i][j] = (float **) malloc(sizeof(float *) * filter_size);
                if ((*weights)[i][j] == NULL) {
                    fprintf(stderr, "Error allocating memory for conv weights.\n");
                    for (int k = 0; k < j; k++) {
                        free((*weights)[i][k]);
                    }
                    free((*weights)[i]);
                    for (int j = 0; j < i; j++) {
                        free((*weights)[j]);
                    }
                    free(*weights);
                    H5Dclose(dataset_id);
                    H5Sclose(dataspace_id);
                    return -1;
                }

                for (int k = 0; k < filter_size; k++) {
                    (*weights)[i][j][k] = (float *) malloc(sizeof(float) * filter_size);
                    if ((*weights)[i][j][k] == NULL) {
                        fprintf(stderr, "Error allocating memory for conv weights.\n");
                        for (int l = 0; l < k; l++) {
                            free((*weights)[i][j][l]);
                        }
                        free((*weights)[i][j]);
                        for (int k = 0; k < j; k++) {
                            free((*weights)[i][k]);
                        }
                        free((*weights)[i]);
                        for (int j = 0; j < i; j++) {
                            free((*weights)[j]);
                        }
                        free(*weights);
                        H5Dclose(dataset_id);
                        H5Sclose(dataspace_id);
                        return -1;
                    }
                }
            }
        }
    }

    herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *weights);
    if (status < 0) {
        fprintf(stderr, "Error reading HDF5 dataset: %s\n", dataset_name);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        return -1;
    }

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    return 0;
}

static herr_t load_fc_weights(hid_t group_id, float ***weights, int num_neurons, int input_size) {
    if (weights == NULL || num_neurons <= 0 || input_size <= 0) {
        fprintf(stderr, "Error: Invalid input parameters for load_fc_weights.\n");
        return -1;
    }

    hid_t dataset_id, dataspace_id;
    char dataset_name[MAX_DATASET_NAME_LEN];

    snprintf(dataset_name, MAX_DATASET_NAME_LEN, "fc_weights");
    dataset_id = H5Dopen(group_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening HDF5 dataset: %s\n", dataset_name);
        return -1;
    }

    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        fprintf(stderr, "Error getting HDF5 dataspace.\n");
        H5Dclose(dataset_id);
        return -1;
    }

    if (*weights == NULL) {
        *weights = (float **) malloc(sizeof(float *) * num_neurons);
        if (*weights == NULL) {
            fprintf(stderr, "Error allocating memory for fc weights.\n");
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
            return -1;
        }

        for (int i = 0; i < num_neurons; i++) {
            (*weights)[i] = (float *) malloc(sizeof(float) * input_size);
            if ((*weights)[i] == NULL) {
                fprintf(stderr, "Error allocating memory for fc weights.\n");
                for (int j = 0; j < i; j++) {
                    free((*weights)[j]);
                }
                free(*weights);
                H5Dclose(dataset_id);
                H5Sclose(dataspace_id);
                return -1;
            }
        }
    }

    herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *weights);
    if (status < 0) {
        fprintf(stderr, "Error reading HDF5 dataset: %s\n", dataset_name);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        return -1;
    }

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    return 0;
}

static herr_t load_biases(hid_t group_id, float **biases, int num_biases) {
    if (biases == NULL || num_biases <= 0) {
        fprintf(stderr, "Error: Invalid input parameters for load_biases.\n");
        return -1;
    }

    hid_t dataset_id, dataspace_id;
    char dataset_name[MAX_DATASET_NAME_LEN];

    snprintf(dataset_name, MAX_DATASET_NAME_LEN, "biases");
    dataset_id = H5Dopen(group_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening HDF5 dataset: %s\n", dataset_name);
        return -1;
    }

    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        fprintf(stderr, "Error getting HDF5 dataspace.\n");
        H5Dclose(dataset_id);
        return -1;
    }

    if (*biases == NULL) {
        *biases = (float *) malloc(sizeof(float) * num_biases);
        if (*biases == NULL) {
            fprintf(stderr, "Error allocating memory for biases.\n");
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
            return -1;
        }
    }

    herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *biases);
    if (status < 0) {
        fprintf(stderr, "Error reading HDF5 dataset: %s\n", dataset_name);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        return -1;
    }

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    return 0;
}

static herr_t load_layer_weights(hid_t group_id, Layer* layer) {
    switch (layer->type) {
        case CONVOLUTIONAL:
            // Load convolutional layer weights and biases
            if (load_conv_weights(group_id, &layer->weights.conv_weights, layer->params.conv_params.num_filters, layer->params.conv_params.filter_size, layer->input_shape.channels) != 0) {
                return -1;
            }
            if (load_biases(group_id, &layer->biases, layer->params.conv_params.num_filters) != 0) {
                return -1;
            }
            break;
        case POOLING:
            // Pooling layer does not have any weights or biases
            break;
        case FULLY_CONNECTED:
            // Load fully connected layer weights and biases
            if (load_fc_weights(group_id, &layer->weights.fc_weights, layer->params.fc_params.num_neurons, layer->input_shape.width * layer->input_shape.height * layer->input_shape.channels) != 0) {
                return -1;
            }
            if (load_biases(group_id, &layer->biases, layer->params.fc_params.num_neurons) != 0) {
                return -1;
            }
            break;
        // Add other layer types as needed
        default:
            fprintf(stderr, "Error: Unsupported layer type.\n");
            return -1;
    }

    return 0;
}

void load_model_weights(Model* model, const char* filename) {
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    if (filename == NULL || strlen(filename) == 0) {
        fprintf(stderr, "Error: Invalid filename.\n");
        return;
    }

    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error opening HDF5 file: %s\n", filename);
        return;
    }

    for (int i = 0; i < model->num_layers; i++) {
        Layer* layer = model->layers[i];
        if (layer == NULL) {
            fprintf(stderr, "Error: Layer %d pointer is NULL.\n", i);
            H5Fclose(file_id);
            return;
        }

        char group_name[MAX_GROUP_NAME_LEN];
        snprintf(group_name, MAX_GROUP_NAME_LEN, "/layer_%d_%s", i, get_layer_type_string(layer->type));

        hid_t group_id = H5Gopen(file_id, group_name, H5P_DEFAULT);
        if (group_id < 0) {
            fprintf(stderr, "Error opening HDF5 group: %s\n", group_name);
            H5Fclose(file_id);
            return;
        }

        herr_t load_status = load_layer_weights(group_id, layer);
        if (load_status != 0) {
            fprintf(stderr, "Error loading weights for layer %d\n", i);
            H5Gclose(group_id);
            H5Fclose(file_id);
            return;
        }

        H5Gclose(group_id);
    }

    H5Fclose(file_id);
}

static int get_vgg16_group_name(Layer* layer, char* group_name, size_t group_name_len, int* conv_block_count, int* fc_layer_count) {
    if (layer == NULL || group_name == NULL || group_name_len == 0 || conv_block_count == NULL || fc_layer_count == NULL) {
        return -1;
    }

    switch (layer->type) {
        case CONVOLUTIONAL:
            if (layer->output_shape.channels == 64) {
                snprintf(group_name, group_name_len, "/conv%d_%d", 1, *conv_block_count);
            } else if (layer->output_shape.channels == 128) {
                snprintf(group_name, group_name_len, "/conv%d_%d", 2, *conv_block_count - 2);
            } else if (layer->output_shape.channels == 256) {
                snprintf(group_name, group_name_len, "/conv%d_%d", 3, *conv_block_count - 4);
            } else if (layer->output_shape.channels == 512 && *conv_block_count <= 10) {
                snprintf(group_name, group_name_len, "/conv%d_%d", 4, *conv_block_count - 7);
            } else if (layer->output_shape.channels == 512 && *conv_block_count > 10) {
                snprintf(group_name, group_name_len, "/conv%d_%d", 5, *conv_block_count - 10);
            }
            (*conv_block_count)++;
            break;
        case FULLY_CONNECTED:
            snprintf(group_name, group_name_len, "/fc%d", *fc_layer_count);
            (*fc_layer_count)++;
            break;
        default:
            return -1;
    }

    return 0;
}

void load_vgg16_weights(Model* model, const char* filename) {
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    if (filename == NULL || strlen(filename) == 0) {
        fprintf(stderr, "Error: Invalid filename.\n");
        return;
    }

    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error opening HDF5 file: %s\n", filename);
        return;
    }

    int conv_block_count = 1;
    int fc_layer_count = 6;

    for (int i = 0; i < model->num_layers; i++) {
        Layer* layer = model->layers[i];
        if (layer == NULL) {
            fprintf(stderr, "Error: Layer %d pointer is NULL.\n", i);
            H5Fclose(file_id);
            return;
        }

        char group_name[MAX_GROUP_NAME_LEN];
        if (get_vgg16_group_name(layer, group_name, MAX_GROUP_NAME_LEN, &conv_block_count, &fc_layer_count) != 0) {
            fprintf(stderr, "Error generating group name for layer %d\n", i);
            H5Fclose(file_id);
            return;
        }

        hid_t group_id = H5Gopen(file_id, group_name, H5P_DEFAULT);
        if (group_id < 0) {
            fprintf(stderr, "Error opening HDF5 group: %s\n", group_name);
            H5Fclose(file_id);
            return;
        }

        herr_t load_status = load_layer_weights(group_id, layer);
        if (load_status != 0) {
            fprintf(stderr, "Error loading weights for layer %d\n", i);
            H5Gclose(group_id);
            H5Fclose(file_id);
            return;
        }

        H5Gclose(group_id);
    }

    H5Fclose(file_id);
}
