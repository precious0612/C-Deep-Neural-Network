/* model/model.c */

#include <stdio.h>
#include <stdlib.h> // for malloc and free
#include <string.h> // for strcmp

#include "model.h"
#include "layer/layer.h"

#define MAX_BATCH_PRINT 100
#define PRINT_INTERVAL(total_batches) ((total_batches) <= MAX_BATCH_PRINT ? 1 : MAX_BATCH_PRINT)

Model* create_model(Dimensions input, Dimensions output) {
    Model* model = (Model*)malloc(sizeof(Model));
    model->input = input;
    model->output = output;
    model->layers = NULL;
    model->num_layers = 0;
    return model;
}

void add_layer_to_model(Model* model, Layer* layer) {
    model->num_layers++;
    model->layers = (Layer**)realloc(model->layers, model->num_layers * sizeof(Layer*));
    model->layers[model->num_layers - 1] = layer;
}

void add_layer(Model* model, char* layer_type, int num_filters, int filter_size, int stride, int padding, char* activation, float dropout_rate) {
    LayerParams params;

    if (strcmp(layer_type, "convolutional") == 0) {
        params.conv_params.num_filters = num_filters;
        params.conv_params.filter_size = filter_size;
        params.conv_params.stride = stride;
        params.conv_params.padding = padding;
        strcpy(params.conv_params.activation, activation);

        Layer* layer = create_layer(CONVOLUTIONAL, params);
        add_layer_to_model(model, layer);
    } else if (strcmp(layer_type, "max_pooling") == 0) {
        params.pooling_params.pool_size = filter_size;
        params.pooling_params.stride = stride;
        strcpy(params.pooling_params.pool_type, "max");

        Layer* layer = create_layer(POOLING, params);
        add_layer_to_model(model, layer);
    } else if (strcmp(layer_type, "fully_connected") == 0) {
        params.fc_params.num_neurons = num_filters;
        strcpy(params.fc_params.activation, activation);

        Layer* layer = create_layer(FULLY_CONNECTED, params);
        add_layer_to_model(model, layer);
    } else if (strcmp(layer_type, "dropout") == 0) {
        params.dropout_params.dropout_rate = dropout_rate;

        Layer* layer = create_layer(DROPOUT, params);
        add_layer_to_model(model, layer);
    } else if (strcmp(layer_type, "activation") == 0) {
        params.activation_params.activation[0] = '\0';
        strcpy(params.activation_params.activation, activation);

        Layer* layer = create_layer(ACTIVATION, params);
        add_layer_to_model(model, layer);
    } else if (strcmp(layer_type, "flatten") == 0) {
        Layer* layer = create_layer(FLATTEN, (LayerParams){});
        add_layer_to_model(model, layer);
    } else {
        fprintf(stderr, "Error: Invalid layer type specified.\n");
    }
}

void set_optimizer(Model* model, const char* optimizer, float learning_rate) {
    strcpy(model->optimizer_name, optimizer);
    model->learning_rate = learning_rate;
}

void set_loss_function(Model* model, const char* loss_function) {
    strcpy(model->loss_function, loss_function);
}

void set_metric(Model* model, const char* metric_name) {
    strcpy(model->metric_name, metric_name);
}

void compile_model(Model* model, char* optimizer_name, float learning_rate, char* loss_function, char* metric_name) {
    // Check if the model pointer is valid
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    // Assign the configuration settings to the model
    strcpy(model->optimizer_name, optimizer_name);
    model->learning_rate = learning_rate;
    strcpy(model->loss_function, loss_function);
    strcpy(model->metric_name, metric_name);

    // Default loss function
    LossFunction loss_fn = mean_squared_error_loss;

    // Assign the loss function
    if (strcmp(loss_function, "categorical_crossentropy") == 0) {
        model->loss_fn = categorical_crossentropy_loss;
    } else if (strcmp(loss_function, "mean_squared_error") == 0) {
        model->loss_fn = mean_squared_error_loss;
    } else {
        // User-provided loss function
        model->loss_fn = loss_fn;
    }

    int num_layers = model->num_layers;
    int* num_weights = malloc(num_layers * sizeof(int));
    if (num_weights == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for num_weights array\n");
        exit(1);
    }

    for (int i = 0; i < num_layers; i++) {
        Layer* current_layer = model->layers[i];
        switch (current_layer->type) {
            case CONVOLUTIONAL:
                num_weights[i] = current_layer->params.conv_params.num_filters * current_layer->params.conv_params.filter_size * current_layer->params.conv_params.filter_size * current_layer->input_shape.channels + current_layer->params.conv_params.num_filters;
                break;
            case FULLY_CONNECTED:
                num_weights[i] = current_layer->params.fc_params.num_neurons * current_layer->input_shape.width * current_layer->input_shape.height * current_layer->input_shape.channels + current_layer->params.fc_params.num_neurons;
                break;
            default:
                num_weights[i] = 0;
                break;
        }
    }

    // Create and initialize the optimizer

    model->optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    if (model->optimizer == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for optimizer\n");
        exit(1);
    }
    if (strcmp(optimizer_name, "sgd") == 0 || strcmp(optimizer_name, "SGD") == 0) {
        model->optimizer->type = SGD;
        model->optimizer->optimizer.sgd = init_sgd(learning_rate, 0.0f, num_weights, num_layers); // Initialize momentum to 0
    } else if (strcmp(optimizer_name, "adam") == 0 || strcmp(optimizer_name, "Adam") == 0) {
        model->optimizer->type = ADAM;
        // Initialize Adam optimizer with default values for beta1, beta2, and epsilon
        // beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8
        model->optimizer->optimizer.adam = init_adam(learning_rate, 0.9, 0.999, 1e-8, num_weights, num_layers);
    } else if (strcmp(optimizer_name, "rmsprop") == 0 || strcmp(optimizer_name, "RMSprop") == 0) {
        model->optimizer->type = RMSPROP;
        // Initialize RMSprop optimizer with default values for rho and epsilon
        // rho = 0.9, epsilon = 1e-8
        model->optimizer->optimizer.rmsprop = init_rmsprop(learning_rate, 0.9, 1e-8, num_weights, num_layers);
    } else {
        fprintf(stderr, "Error: Invalid optimizer.\n");
        free(num_weights);
        return;
    }

    free(num_weights); // Free the num_weights array

    // Set the input shape of the first layer
    if (model->num_layers > 0) {
        model->layers[0]->input_shape.width = model->input.width;
        model->layers[0]->input_shape.height = model->input.height;
        model->layers[0]->input_shape.channels = model->input.channels;
    } else {
        fprintf(stderr, "Error: Model does not have any layers.\n");
        delete_optimizer(model->optimizer, num_layers);
        return;
    }

    Layer* prev_layer = NULL;
    // Compute the output shapes for each layer
    for (int i = 0; i < model->num_layers; i++) {
        Layer* current_layer = model->layers[i];
        current_layer->prev_layer = prev_layer;
        prev_layer = current_layer;
        current_layer->next_layer = (i < model->num_layers - 1) ? model->layers[i + 1] : NULL;
        initialize_layer(current_layer);
        compute_output_shape(current_layer);
        if (i < model->num_layers - 1) {
            model->layers[i + 1]->input_shape = current_layer->output_shape;
        }
    }

    // Check if the final layer output shape matches the output information
    Layer* final_layer = model->layers[model->num_layers - 1];
    if (final_layer->output_shape.width == model->output.width &&
        final_layer->output_shape.height == model->output.height &&
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
    }
}

float*** forward_pass(Model* model, float*** input) {
    float*** output;

    if (model->num_layers == 0) {
        fprintf(stderr, "Error: Model does not have any layers.\n");
        return copy_3d_array(input, model->input);
    } else {
        output = layer_forward_pass(model->layers[0], input);
    }

    for (int i = 1; i < model->num_layers; i++) {
        output = layer_forward_pass(model->layers[i], output);
    }

    return output;
}


void backward_pass(Model* model, float*** input, float*** output_grad) {
    float*** temp_input_grad = output_grad;
    for (int i = model->num_layers - 1; i >= 0; i--) {
        float*** layer_input_grad = allocate_grad_tensor(model->layers[i]->input_shape);
        layer_backward_pass(model->layers[i], input, temp_input_grad, layer_input_grad);
        temp_input_grad = layer_input_grad;
    }

    // // Update weights and biases for all layers
    // for (int i = 0; i < model->num_layers; i++) {
    //     update_layer_weights(model->layers[i], learning_rate);
    // }
}

float**** forward_pass_batch(Model* model, float**** batch_inputs, int batch_size) {
    float ****batch_outputs = (float****)malloc(sizeof(float***) * batch_size);
    for (int i = 0; i < batch_size; i++) {
        batch_outputs[i] = forward_pass(model, batch_inputs[i]);
    }
    return batch_outputs;
}

void backward_pass_batch(Model* model, float**** batch_inputs, float**** batch_output_grads, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        backward_pass(model, batch_inputs[i], batch_output_grads[i]);
    }
}

void update_model_weights(Model* model) {
    for (int i = 0; i < model->num_layers; i++) {
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

void train_model(Model* model, Dataset* dataset, int num_epochs) {
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        Dataset* batch = dataset;
        int batch_count = 0;
        int total_batches = 0;

        // Calculate the total number of batches
        while (batch != NULL) {
            total_batches++;
            batch = batch->next_batch;
        }

        int print_interval = PRINT_INTERVAL(total_batches);

        batch = dataset;
        while (batch != NULL) {
            int batch_num = batch_count++;
            float**** batch_inputs = (float****)malloc(sizeof(float***) * batch->batch_size);
            float**** batch_outputs;
            int* batch_labels = batch->labels;

            for (int i = 0; i < batch->batch_size; i++) {
                switch (batch->data_type) {
                    case Int:
                        batch_inputs[i] = batch->images[i]->int_data;
                        break;
                    case FLOAT32:
                        batch_inputs[i] = batch->images[i]->float32_data;
                        break;
                    default:
                        fprintf(stderr, "Error: Invalid data type specified.\n");
                        // Free the allocated memory before returning
                        for (int j = 0; j < i; j++) {
                            free_tensor(batch_outputs[j], model->output);
                        }
                        free(batch_inputs);
                        free(batch_outputs);
                        return;
                }
                // batch_outputs[i] = allocate_output_tensor(model->output);
            }

            float**** batch_output_grads = (float****)malloc(sizeof(float***) * batch->batch_size);
            for (int i = 0; i < batch->batch_size; i++) {
                batch_output_grads[i] = allocate_grad_tensor(model->output);
            }

            batch_outputs = forward_pass_batch(model, batch_inputs, batch->batch_size);
            compute_output_grad_batch(batch_outputs, batch_labels, batch_output_grads, batch->batch_size, model->output.height, model->output.width, model->output.channels);
            backward_pass_batch(model, batch_inputs, batch_output_grads, batch->batch_size);
            update_model_weights(model);
            // reset_model_grads(model);  // reset operation has been finished at the beginning of the backward pass in each layer

            if (total_batches <= 100 || (batch_num + 1) % print_interval == 0 || batch_num == total_batches - 1) {
                // Print progress
                printf("\nEpoch %d, Batch %d/%d, DataNum %d\n", epoch + 1, batch_num + 1, total_batches, batch->num_images);

                // Print loss and accuracy for the batch
                float loss = compute_loss_batch(batch_outputs, batch_labels, model->loss_fn, batch->batch_size, model->output.channels);
                float accuracy = compute_accuracy(batch_outputs, batch_labels, batch->batch_size, model->output.channels);
                printf("Epoch %d, Batch Loss: %.6f, Batch Accuracy: %.2f%%\n", epoch + 1, loss, accuracy * 100.0f);
            }

            // Free the allocated memory for batch_outputs and batch_output_grads
            for (int i = 0; i < batch->batch_size; i++) {
                free_tensor(batch_outputs[i], model->output);
                free_tensor(batch_output_grads[i], model->output);
            }

            free(batch_inputs);
            free(batch_outputs);
            free(batch_output_grads);

            // Print progress bar for the current epoch
            float epoch_progress = (float)(batch_count) / total_batches;
            print_progress_bar(epoch_progress);

            batch = batch->next_batch;
        }
        // Print progress
        printf("\nEpoch %d completed.\n", epoch + 1);

        // Evaluate the model on the validation dataset
        if (dataset->val_dataset != NULL) {
            float val_accuracy = evaluate_model(model, dataset->val_dataset);
            printf("\nValidation Accuracy: %.2f%%\n", val_accuracy * 100.0f);
        }
    }
}

float evaluate_model(Model* model, Dataset* dataset) {
    int correct_predictions = 0;
    int total_samples = 0;

    Dataset* batch = dataset;
    int batch_count = 0;
    int total_batches = 0;

    // Calculate the total number of batches
    while (batch != NULL) {
        total_batches++;
        batch = batch->next_batch;
    }

    batch = dataset;
    while (batch != NULL) {
        printf("Evaluating Batch %d/%d\n", batch_count + 1, total_batches);

        for (int i = 0; i < batch->num_images; i++) {
            // preprocess_input(batch->images[i], model->input);
            // float*** output = allocate_output_tensor(model->output);
            float*** output;

            switch (batch->data_type) {
                case Int:
                    output = forward_pass(model, batch->images[i]->int_data);
                    break;
                case FLOAT32:
                    output = forward_pass(model, batch->images[i]->float32_data);
                    break;
                default:
                    fprintf(stderr, "Error: Invalid data type specified.\n");
                    return 0.0f;
            }

            int prediction = get_prediction(output, model->metric_name, model->output.channels);
            if (prediction == batch->labels[i]) {
                correct_predictions++;
            }
            total_samples++;

            free_tensor(output, model->output);
        }

        // Print progress bar for the current batch
        float batch_progress = (float)(batch_count + 1) / total_batches;
        print_progress_bar(batch_progress);

        batch_count++;
        batch = batch->next_batch;
    }

    printf("\n");
    return (float)correct_predictions / (float)total_samples;
}

void print_model_info(Model* model) {
    printf("Model Configuration:\n");
    printf("Input Shape: (%d, %d, %d)\n", model->input.width, model->input.height, model->input.channels);
    printf("Optimizer: %s\n", model->optimizer_name);
    printf("Learning Rate: %.6f\n", model->learning_rate);
    printf("Loss Function: %s\n", model->loss_function);
    printf("Evaluation Metric: %s\n", model->metric_name);

    // Iterate through layers and print their types and parameters
    int layer_num = 1;
    for (int i = 0; i < model->num_layers; i++) {
        Layer* current_layer = model->layers[i];
        printf("\nLayer %d: ", layer_num);
        switch (current_layer->type) {
            case CONVOLUTIONAL:
                printf("Convolutional\n");
                printf("  Input Shape: (%d, %d, %d)\n", current_layer->input_shape.width, current_layer->input_shape.height, current_layer->input_shape.channels);
                printf("  Number of Filters: %d\n", current_layer->params.conv_params.num_filters);
                printf("  Filter Size: %d\n", current_layer->params.conv_params.filter_size);
                printf("  Stride: %d\n", current_layer->params.conv_params.stride);
                printf("  Padding: %d\n", current_layer->params.conv_params.padding);
                printf("  Output Shape: (%d, %d, %d)\n", current_layer->output_shape.width, current_layer->output_shape.height, current_layer->output_shape.channels);
                printf("  Activation Function: %s\n", current_layer->params.conv_params.activation);
                break;
            case POOLING:
                printf("Pooling\n");
                printf("  Input Shape: (%d, %d, %d)\n", current_layer->input_shape.width, current_layer->input_shape.height, current_layer->input_shape.channels);
                printf("  Pool Size: %d\n", current_layer->params.pooling_params.pool_size);
                printf("  Stride: %d\n", current_layer->params.pooling_params.stride);
                printf("  Pool Type: %s\n", current_layer->params.pooling_params.pool_type);
                printf("  Output Shape: (%d, %d, %d)\n", current_layer->output_shape.width, current_layer->output_shape.height, current_layer->output_shape.channels);
                break;
            case FULLY_CONNECTED:
                printf("Fully Connected\n");
                printf("  Input Shape: (%d, %d, %d)\n", current_layer->input_shape.width, current_layer->input_shape.height, current_layer->input_shape.channels);
                printf("  Number of Neurons: %d\n", current_layer->params.fc_params.num_neurons);
                printf("  Activation Function: %s\n", current_layer->params.fc_params.activation);
                printf("  Output Shape: (%d, %d, %d)\n", current_layer->output_shape.width, current_layer->output_shape.height, current_layer->output_shape.channels);
                break;
            case DROPOUT:
                printf("Dropout\n");
                printf("  Input Shape: (%d, %d, %d)\n", current_layer->input_shape.width, current_layer->input_shape.height, current_layer->input_shape.channels);
                printf("  Dropout Rate: %.2f\n", current_layer->params.dropout_params.dropout_rate);
            case ACTIVATION:
                printf("Activation\n");
                printf("  Activation Function: %s\n", current_layer->params.activation_params.activation);
                break;
            case FLATTEN:
                printf("Flatten\n");
                printf("  Input Shape: (%d, %d, %d)\n", current_layer->input_shape.width, current_layer->input_shape.height, current_layer->input_shape.channels);
                printf("  Output Shape: (%d, %d, %d)\n", current_layer->output_shape.width, current_layer->output_shape.height, current_layer->output_shape.channels);
                break;
            // Add cases for other layer types as needed
        }
        layer_num++;
    }
}

void delete_model(Model* model) {
    // Check if the model pointer is valid
    if (model == NULL) {
        fprintf(stderr, "Error: Model pointer is NULL.\n");
        return;
    }

    for (int i = 0; i < model->num_layers; i++) {
        delete_layer(model->layers[i]);
    }
    free(model->layers);
    free(model);
}
