# C Deep Neural Network (cdnn) API

## NOTICE

The current interface only supports multi-core CPU computing on different platforms (based on `openBLAS` interface), and support for graphics computation acceleration on different platforms will be added later. (Nvidia's `CUDA` is expected to be supported first because it provides a C interface, while `Apple Silicon` currently only offers a C++ interface.)

**This project is currently being developed by me alone out of personal interest, and I am just a recent college graduate. Therefore, there may be many issues with this project. I hope more interested people will join me.**

This API is still under development and is not yet ready for production use. Please use it with caution and report any issues or bugs you encounter.

## An API Test Usage

How to run the test example? (After cloning the project...)

```shell
make clean && make && make run_example
```
**(Notice: You need to install `libpng`, `jpeg-turbo`, `stb_image`, `openblas`, `hdf5`, `json-c`)**

## Project Introduction

This project aims to provide a high-level API for designing, training, and evaluating Deep Neural Network models in C. The API is designed to be user-friendly and easy to integrate into existing C projects. It includes functions for creating models, adding layers, compiling models, training models, evaluating models, and saving/loading models and weights (with `HDF5` Format).

### Code Example:

```c
#include "CNN.h"
#include "dataset.h"

int main() {
    // Create a new CNN model
    Model* model = create(28, 28, 1, 1, 1, 10);

    // Add layers
    add_convolutional_layer(model, 32, 3, 1, 1, "relu");
    add_max_pooling_layer(model, 2, 2);
    add_fully_connected_layer(model, 128, "relu");
    add_dropout_layer(model, 0.5);
    add_fully_connected_layer(model, 10, "softmax");

    // Compile the model
    ModelConfig config = {"Adam", 0.001f, "categorical_crossentropy", "accuracy"};
    compile(model, config);

    // Load datasets
    Dataset* train_dataset = load_dataset("train_data.json");
    Dataset* test_dataset = load_dataset("test_data.json");

    // Train the model
    train(model, train_dataset, 10);

    // Evaluate the model
    float accuracy = evaluate(model, test_dataset);
    printf("Accuracy: %.2f%%\n", accuracy * 100);

    // Free memory
    free_model(model);
    free_dataset(train_dataset);
    free_dataset(test_dataset);

    return 0;
}
```

## Standard JSON Dataset Format of this API

### Model Config

```json
{
    "input_shape": [28, 28, 1],
    "output_shape": [10],
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss_function": "categorical_crossentropy",
    "metric_name": "accuracy",
    "layers": [
        {
            "type": "convolutional",
            "params": {
                "num_filters": 32,
                "filter_size": 3,
                "stride": 1,
                "padding": "same",
                "activation": "relu"
            }
        },
        {
            "type": "pooling",
            "params": {
                "pool_size": 2,
                "stride": 2,
                "pool_type": "max"
            }
        },
        {
            "type": "fully_connected",
            "params": {
                "num_neurons": 10,
                "activation": "relu"
            }
        }
    ]
}
```

### Dataset with val folder

```json
{
  "dataset_name": "test_data_and_val",
  "num_images": 2,
  "images": [
    {
      "file_path": "dataset example/test_data_and_val/0/test.jpeg",
      "label": "0"
    }
,    {
      "file_path": "dataset example/test_data_and_val/1/test.png",
      "label": "1"
    }
  ]
,
  "val_dataset": "val",
  "num_val_images": 2,
  "val_images": [
    {
      "file_path": "dataset example/test_data_and_val/val/0/test.jpeg",
      "label": "0"
    }
,    {
      "file_path": "dataset example/test_data_and_val/val/1/test.png",
      "label": "1"
    }
  ]
}
```

### Dataset without val folder

```json
{
  "dataset_name": "test_data_without_val",
  "num_images": 4,
  "images": [
    {
      "file_path": "dataset example/test_data_without_val/0/test.jpeg",
      "label": "0"
    }
,    {
      "file_path": "dataset example/test_data_without_val/0/IMG_0255_smaller.jpeg",
      "label": "0"
    }
,    {
      "file_path": "dataset example/test_data_without_val/1/test.png",
      "label": "1"
    }
,    {
      "file_path": "dataset example/test_data_without_val/1/1528_1307.png",
      "label": "1"
    }
  ]
,
  "validation_size": 0.50
}
```

## External Libraries

- [libpng](https://github.com/glennrp/libpng): Library for reading and writing PNG image files.
  License: [PNG Reference Library License](http://www.libpng.org/pub/png/src/libpng-LICENSE.txt)

- [jpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo): Library for reading and writing JPEG image files. 
  License: [IJG License](https://www.ijg.org/files/README)

- [json-c](https://github.com/json-c/json-c): Library for parsing JSON data. 
  License: [MIT License](https://opensource.org/licenses/MIT)

- [hdf5](https://www.hdfgroup.org/solutions/hdf5/): Library for reading and writing HDF5 files.
  License: [HDF5 License](https://github.com/HDFGroup/hdf5?tab=License-1-ov-file#)
