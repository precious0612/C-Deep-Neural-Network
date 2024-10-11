#include <stdio.h>
#include "../dataset.h"

int main(int argc, char* argv[]) {
    // if (argc != 5) {
    //     fprintf(stderr, "Usage: %s <train_images> <train_labels> <test_images> <test_labels>\n", argv[0]);
    //     return 1;
    // }

    const char* train_images_path = "/Users/precious/Design_Neural_Network/dataset example/mnist/train-images-idx3-ubyte.gz";
    const char* train_labels_path = "/Users/precious/Design_Neural_Network/dataset example/mnist/train-labels-idx1-ubyte.gz";
    const char* test_images_path = "/Users/precious/Design_Neural_Network/dataset example/mnist/t10k-images-idx3-ubyte.gz";
    const char* test_labels_path = "/Users/precious/Design_Neural_Network/dataset example/mnist/t10k-labels-idx1-ubyte.gz";

    Dataset* dataset = load_mnist_dataset(train_images_path, train_labels_path,
                                           test_images_path, test_labels_path, FLOAT32);
    if (dataset == NULL) {
        fprintf(stderr, "Error: Failed to load MNIST dataset\n");
        return 1;
    }

    printf("Loaded MNIST dataset:\n");
    printf("  Name: %s\n", dataset->name);
    printf("  Dimensions: %d x %d x %d\n", dataset->data_dimensions.height, dataset->data_dimensions.width, dataset->data_dimensions.channels);
    printf("  Data Type: %s\n", dataset->data_type == SINT32? "UINT8" : "FLOAT32");
    printf("  Training Set Size: %d\n", dataset->num_images);
    printf("  Test Set Size: %d\n", dataset->val_dataset->num_images);

    // Free the dataset
    free_dataset(dataset);

    return 0;
}