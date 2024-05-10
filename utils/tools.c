/* utils/tools.c */

#include <string.h>
#include <stdio.h>
#include "tools.h"

#define PROGRESS_BAR_LENGTH 50

int is_empty_string(const char* str) {
    return str == NULL || str[0] == '\0';
}

int not_empty_string(const char* str) {
    return str != NULL && str[0] != '\0';
}

void print_progress_bar(float progress) {
    int filled_length = (int)(progress * PROGRESS_BAR_LENGTH);
    int remaining_length = PROGRESS_BAR_LENGTH - filled_length;

    printf("\r[");
    for (int i = 0; i < filled_length; i++) {
        printf("#");
    }
    for (int i = 0; i < remaining_length; i++) {
        printf("-");
    }
    printf("] %3.0f%%", progress * 100);
    fflush(stdout);

    if (progress >= 1.0) {
        printf("\n");
    }
}

void save_conv_weights(hid_t group_id, float ****weights, int num_filters, int filter_size, int channels) {
    hid_t dataset_id, dataspace_id;
    hsize_t dims[4] = {num_filters, channels, filter_size, filter_size};

    dataspace_id = H5Screate_simple(4, dims, NULL);
    dataset_id = H5Dcreate(group_id, "weight", H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *weights);

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
}

void save_fc_weights(hid_t group_id, float **weights, int num_neurons, int input_size) {
    hid_t dataset_id, dataspace_id;
    hsize_t dims[2] = {num_neurons, input_size};

    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(group_id, "weight", H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *weights);

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
}

void save_biases(hid_t group_id, float *biases, int num_biases) {
    hid_t dataset_id, dataspace_id;
    hsize_t dims[1] = {num_biases};

    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(group_id, "bias", H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, biases);

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
}

void load_conv_weights(hid_t group_id, float *****weights, int num_filters, int filter_size, int channels) {
    hid_t dataset_id, dataspace_id;
    hsize_t dims[4] = {num_filters, channels, filter_size, filter_size};

    dataset_id = H5Dopen(group_id, "weight", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);

    if (*weights != NULL) {
        *weights = (float ****) malloc(sizeof(float ***) * num_filters);
        for (int i = 0; i < num_filters; i++) {
            (*weights)[i] = (float ***) malloc(sizeof(float **) * channels);
            for (int j = 0; j < channels; j++) {
                (*weights)[i][j] = (float **) malloc(sizeof(float *) * filter_size);
                for (int k = 0; k < filter_size; k++) {
                    (*weights)[i][j][k] = (float *) malloc(sizeof(float) * filter_size);
                }
            }
        }
    }
 
    float* temp_weights = (float *)malloc(sizeof(float) * num_filters * channels * filter_size * filter_size);

    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_weights);
    for (int i = 0; i < num_filters; i++) {
        for (int j = 0; j < channels; j++) {
            for (int k = 0; k < filter_size; k++) {
                for (int l = 0; l < filter_size; l++) {
                    (*weights)[i][j][k][l] = temp_weights[i * channels * filter_size * filter_size + j * filter_size * filter_size + k * filter_size + l];
                }
            }
        }
    }

    free(temp_weights);

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
}

void load_fc_weights(hid_t group_id, float ***weights, int num_neurons, int input_size) {
    hid_t dataset_id, dataspace_id;
    hsize_t dims[2] = {num_neurons, input_size};

    dataset_id = H5Dopen(group_id, "weight", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);

    if (*weights == NULL) {
        *weights = (float **) malloc(sizeof(float *) * num_neurons);
        for (int i = 0; i < num_neurons; i++) {
            (*weights)[i] = (float *) malloc(sizeof(float) * input_size);
        }
    }

    float* temp_weights = (float *)malloc(sizeof(float) * num_neurons * input_size);

    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_weights);

    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < input_size; j++) {
            (*weights)[i][j] = temp_weights[i * input_size + j];
        }
    }

    free(temp_weights);

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
}

void load_biases(hid_t group_id, float **biases, int num_biases) {
    hid_t dataset_id, dataspace_id;
    hsize_t dims[1] = {num_biases};

    dataset_id = H5Dopen(group_id, "bias", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);

    if (*biases == NULL) {
        *biases = (float *) malloc(sizeof(float) * num_biases);
    }

    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *biases);

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
}
