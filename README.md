# Convolutional Neural Network (CNN) API

## 项目简介 (Project Introduction)

- The aim of this design is to create an open-source machine learning algorithm library encapsulated in the C language. It will offer a comprehensive range of algorithms and interfaces tailored to the diverse needs of machine learning development across various contexts. The significance of this endeavor is multifold:

  - Facilitating machine learning developers with a convenient, efficient, and extensible codebase in C. This empowers them to directly implement a wide array of machine learning tasks, spanning classification, regression, clustering, recommendation systems, image processing, natural language processing, and more. By eliminating the need for additional languages or frameworks, developers can streamline their workflows.

  - Providing machine learning developers with an adaptable, compatible, and high-performance codebase suitable for deployment across diverse devices. This ensures that developers can seamlessly develop and deploy machine learning applications using C across a spectrum of hardware platforms and operating systems. Whether it's embedded systems, IoT devices, mobile platforms, or cloud servers, the library enables consistent performance and functionality.

  - Offering a concise, lucid, and accessible codebase for machine learning education and research. By providing clear and easily understandable code, learners and researchers can delve into the principles, methodologies, technologies, and applications of machine learning. Additionally, they can gain insights into the advantages and limitations of employing the C language within the realm of machine learning, fostering a deeper understanding and mastery of the field.

- Design Concept Documentation:
  - [Designing the abstract data structure for a Convol.md](Designing%20the%20abstract%20data%20structure%20for%20a%20Convol.md)
  - [Defining the high-level architecture of CNN.md](Defining%20the%20high-level%20architecture%20of%20CNN.md)
  - [Input Data Representation.md](Input%20Data%20Representation.md)

中文：
- 该设计的目的是开源出一个基于C语言所封装的机器学习算法库，提供各类算法与接口，以满足不同场景和需求的机器学习开发。此次设计的意义在于：  
  - 为C语言的机器学习开发者提供一个方便，高效，可扩展的代码库，使得他们可以在不依赖其他语言或框架的情况下，使用C语言直接实现各种机器学习任务，如分类，回归，聚类，推荐，图像处理，自然语言处理等。 
  - 为各种设备的机器学习开发者提供一个适应性强，兼容性好，性能优的代码库，使得他们可以在不同的硬件平台和操作系统上，使用C语言开发和部署机器学习应用，如嵌入式系统，物联网设备，移动设备，云端服务器等。 
  - 为机器学习的教育和研究提供一个简洁，清晰，易懂的代码库，使得学习者和研究者可以通过C语言，深入理解和掌握机器学习的原理，方法，技术和应用，以及C语言在机器学习领域的优势和局限。
- 设计思路说明文档：
  - [Designing the abstract data structure for a Convol.md](Designing%20the%20abstract%20data%20structure%20for%20a%20Convol.md)
  - [Defining the high-level architecture of CNN.md](Defining%20the%20high-level%20architecture%20of%20CNN.md)
  - [Input Data Representation.md](Input%20Data%20Representation.md)

## API 使用方法 (API Usage)

This API provides a high-level interface for model designers to easily define, compile, train, and evaluate CNN models. To correctly utilize this API, please follow the instructions below:

1. Create a new CNN model using the `create` function by specifying the input and output shapes. Alternatively, use the `create_model_from_json` function to load a predefined model configuration and architecture from a JSON file.

2. Add various layers to the model using the following functions:
   - `add_convolutional_layer`: Add a convolutional layer
   - `add_max_pooling_layer`: Add a max pooling layer
   - `add_fully_connected_layer`: Add a fully connected layer
   - `add_dropout_layer`: Add a dropout layer
   - `add_flatten_layer`: Add a flatten layer
   - `add_softmax_layer`: Add a Softmax layer
   - `add_relu_layer`: Add a ReLU layer
   - `add_sigmoid_layer`: Add a Sigmoid layer
   - `add_tanh_layer`: Add a Tanh layer

3. Use the `compile` function to assign the model's configuration, including the optimizer, learning rate, loss function, and evaluation metric.

4. Use the `train` function to train the model on the provided dataset, specifying the number of epochs.

5. Use the `evaluate` function to evaluate the model's performance on the provided dataset.

6. Use the `predict` function to make predictions on given input data.

7. Use the `save_model_to_json` function to save the model configuration and architecture to a JSON file.

8. Use the `free_model` function to release the memory allocated for the model.

9. Use the `save_weights` function to save the model weights to an HDF5 format file.

10. Use the `load_weights` function to load model weights from an HDF5 format file.

11. Use the `load_vgg16` function to load a pre-trained VGG16 CNN model.

For detailed usage instructions and function documentation, please refer to [`CNN.h`](CNN.h).

### 中文

该 API 提供了高级接口，使模型设计人员可以方便地定义、编译、训练和评估 CNN 模型。要正确使用该 API，请按照以下说明进行操作:

1. 使用 `create` 函数创建一个新的 CNN 模型，并指定输入和输出的形状。或者，使用 `create_model_from_json` 函数从 JSON 文件中加载预定义的模型配置和架构。

2. 使用以下函数向模型添加各种层:
   - `add_convolutional_layer`: 添加卷积层
   - `add_max_pooling_layer`: 添加最大池化层
   - `add_fully_connected_layer`: 添加全连接层
   - `add_dropout_layer`: 添加丢弃层
   - `add_flatten_layer`: 添加扁平化层
   - `add_softmax_layer`: 添加 Softmax 层
   - `add_relu_layer`: 添加 ReLU 层
   - `add_sigmoid_layer`: 添加 Sigmoid 层
   - `add_tanh_layer`: 添加 Tanh 层

3. 使用 `compile` 函数为模型分配优化器、学习率、损失函数和评估指标等配置。

4. 使用 `train` 函数在提供的数据集上训练模型，指定epochs数。

5. 使用 `evaluate` 函数在提供的数据集上评估模型的性能。

6. 使用 `predict` 函数对给定的输入数据进行预测。

7. 使用 `save_model_to_json` 函数将模型配置和架构保存到 JSON 文件中。

8. 使用 `free_model` 函数释放为模型分配的内存。

9. 使用 `save_weights` 函数将模型权重保存到 HDF5 格式的文件中。

10. 使用 `load_weights` 函数从 HDF5 格式的文件中加载模型权重。

11. 使用 `load_vgg16` 函数加载预训练的 VGG16 CNN 模型。

有关详细的使用说明和函数文档，请参阅 [`CNN.h`](CNN.h)。

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

## API文件 (API Files)

- [CNN.h](CNN.h): Header file containing function declarations and documentation for the CNN API.
- [dataset.h](input/dataset.h): Header file containing function declarations and documentation for dataset creation and management.
  - [input/test_pic](input/test_pic): Include two demo dataset format and standard .json files 

- [`CNN.h`](CNN.h): 头文件，包含 CNN API 的函数声明和文档。
- [`dataset.h`](input/dataset.h): 头文件，包含数据集创建和管理的函数声明和文档。
  - [`dataset example`](dataset%20example): 包含两个演示数据集格式和标准 `.json` 文件 

## 标准JSON数据集格式 (Standard JSON Dataset Format)

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

## 代码文件 (Code Files)

- [`CNN.h`](CNN.h):
  Provides high-level interface functions for interacting with the CNN model. This header file includes functions for creating, compiling, training, testing, and freeing the CNN model.
  - Defines functions such as `create_model` to initialize a new CNN model with specified input and output dimensions, `add_layer` to append layers to the model, `compile_model` to assign configuration settings to the model, `train_model` to train the model with provided training data, and `test_model` to evaluate the model with test data.
  - Additionally, includes `check_output_shape` to verify if the final layer output shape matches the output information, and `free_model` to release memory allocated for the model.

- [`CNN.c`](CNN.c):
  Provides implementations for the functions defined in `CNN.h`.

- [`dataset.h`](dataset.h):
  Provides definitions and functions for managing datasets used in the CNN API.
  - Defines a `Dataset` structure to describe datasets, including metadata such as name, batch size, and number of images, along with pointers to input data and labels.
  - Offers functions for loading datasets from JSON files, creating JSON files from datasets, creating batches from datasets, splitting datasets into batches, and freeing memory allocated for datasets.

- [`dataset.c`](dataset.c):
  Provides implementations for functions to load datasets from JSON files, create JSON files from datasets, split datasets into batches, and free memory allocated for datasets.

- [`input/data.h`](input/data.h):
  Defines structures and functions for handling input data, including images, for `dataset.h`.
  - Defines a `DataType` enum to specify the data type of input data.
  - Defines an `InputData` union to hold input data in either int or float format.
  - Provides functions for loading input data from images (`load_input_data_from_image` and `load_image_data_with_format`) with support for various image formats (JPEG, PNG, etc.).
  - Provides functions (`loadFloatJPEG`, `loadIntJPEG`, `loadFloatPNG`, `loadIntPNG`, `loadFloatImage`, `loadIntImage`) for loading images from disk in either float or int format for specific image formats.
  - Includes functions to create empty input data (`create_empty_input_data`), resize images (`resize_image`), and free memory associated with input data (`free_image_data`).

- [`input/data.c`](input/data.c):
  Provides implementations for functions to load input data from images, resize images, and free memory associated with input data.

- [`model/layer/layer.h`](model/layer/layer.h):
  Defines the interface for the layer object in the CNN model. This header file includes definitions for layer types, layer parameter structures, weight and bias structures, as well as function declarations for creating, initializing, performing forward and backward passes, updating parameters, and deleting layers.
  - Defines an `LayerType` enum for different types of layers: convolutional, pooling, fully connected, and dropout.
  - Defines a `LayerParams` union to hold parameters specific to each layer type.
  - Defines a `Layer` struct representing a layer in the CNN model, including its type, parameters, weights, biases, and a pointer to the next layer in the model.

- [`model/layer/layer.c`](model/layer/layer.c): 
  Provides implementations for the functions defined in `model/layer/layer.h`, including creating layers, initializing layers, performing forward and backward passes, updating parameters, resetting gradients, computing output shapes, and deleting layers.

- [`model/model.h`](model/model.h):
  Defines the interface for the model object in the CNN. The model object contains methods to perform the forward pass, backward pass, and parameter update for the entire model.
  - Defines a `ModelConfig` struct to hold configuration settings for the model, including optimizer, learning rate, loss function, and evaluation metric.
  - Defines a `Model` struct representing the entire model, including input and output dimensions, optimizer, learning rate, loss function, evaluation metric, and a pointer to the first layer in the model.
  - Provides function declarations for performing the forward pass (`forward_pass`), backward pass (`backward_pass`), training a single epoch (`train_epoch`), and evaluating the model (`evaluate_model`).

- [`model/model.c`](model/model.c):
  Provides implementations for the functions defined in `model/model.h`.

- [`optimizer/optimizer.h`](optimizer/optimizer.h):
  Defines the interface for different optimization algorithms used during the training process.
  - Defines an `OptimizerType` enum to represent different optimizer types (SGD, Adam, RMSprop).
  - Defines separate structs for each optimizer type (`SGDOptimizer`, `AdamOptimizer`, `RMSpropOptimizer`) to hold the respective parameters and state variables.
  - Defines an `Optimizer` struct to hold the optimizer type and a union of the different optimizer structs.
  - Provides function declarations for initializing different optimizer types (`init_sgd`, `init_adam`, `init_rmsprop`), creating an Optimizer instance (`create_optimizer`), and deallocating memory used by the optimizer (`delete_optimizer`).

- [`optimizer/optimizer.c`](optimizer/optimizer.c):
  Provides implementations for the functions defined in `optimizer/optimizer.h`.

- [`utils/compute/activation.h`](utils/compute/activation.h):
  Provides implementations for various activation functions used in neural networks.
  - Defines functions for forward and backward propagation through different activation functions, including ReLU, Sigmoid, Tanh, Max, and Softmax.
  - Includes a unified interface for applying activation functions during the forward pass (`apply_activation`).
  - Includes a unified interface for computing gradients during the backward pass (`apply_activation_backward`).

- [`utils/compute/activation.c`](utils/compute/activation.c):
  Provides implementations for the functions defined in `utils/compute/activation.h`.

- [`utils/compute/convolutional.h`](utils/compute/convolutional.h):
  Provides implementations for performing convolutional operations in Convolutional Neural Networks (CNNs).
  - Defines functions for forward and backward propagation through convolutional layers (`conv_forward`, `conv_backward`).
  - Defines a function for updating the weights and biases of a convolutional layer using an optimization algorithm (`update_conv_weights`).

- [`utils/compute/convolutional.c`](utils/compute/convolutional.c):
  Provides implementations for the functions defined in `utils/compute/convolutional.h`.

- [`utils/compute/dropout.h`](utils/compute/dropout.h):
  Provides implementations for the dropout regularization technique used in neural networks.
  - Defines a function for the forward pass through a dropout layer (`dropout_forward`).
  - Defines a function for the backward pass through a dropout layer (`dropout_backward`).

- [`utils/compute/dropout.c`](utils/compute/dropout.c):
  Provides implementations for the functions defined in `utils/compute/dropout.h`.

- [`utils/compute/flatten.h`](utils/compute/flatten.h):
  Provides implementations for the flatten and unflatten operations used in neural networks.
  - Defines a function for flattening a multi-dimensional tensor into a one-dimensional vector (`flatten`).
  - Defines a function for unflattening a one-dimensional vector into a multi-dimensional tensor (`unflatten`).

- [`utils/compute/flatten.c`](utils/compute/flatten.c):
  Provides implementations for the functions defined in `utils/compute/flatten.h`.

- [`utils/compute/fully_connected.h`](utils/compute/fully_connected.h):
  Provides implementations for fully connected layers used in neural networks.
  - Defines a function for the forward pass through a fully connected layer (`fc_forward`).
  - Defines a function for the backward pass through a fully connected layer (`fc_backward`).
  - Defines a function for updating the weights and biases of a fully connected layer using an optimization algorithm (`update_fc_weights`).

- [`utils/compute/fully_connected.c`](utils/compute/fully_connected.c):
  Provides implementations for the functions defined in `utils/compute/fully_connected.h`.

- [`utils/compute/pooling.h`](utils/compute/pooling.h):
  Provides implementations for pooling operations used in Convolutional Neural Networks (CNNs).
  - Defines a function for the forward pass through a pooling layer (`pool_forward`).
  - Defines a function for the backward pass through a pooling layer (`pool_backward`).

- [`utils/compute/pooling.c`](utils/compute/pooling.c):
  Provides implementations for the functions defined in `utils/compute/pooling.h`.

- [`utils/dimension.h`](utils/dimension.h):
  Defines dimension-related structures and functions.

- [`utils/loss.h`](utils/loss.h):
  Provides implementations for various loss functions used in deep learning.
  - Defines the `LossFunction` type as a function pointer.
  - Includes implementations for Categorical Cross-Entropy Loss and Mean Squared Error Loss.
  - Allows for the integration of custom loss functions following the `LossFunction` signature.

- [`utils/loss.c`](utils/loss.c):
  Provides implementations for the functions defined in `utils/loss.h`.

- [`utils/memory.h`](utils/memory.h):
  Provides utility functions for dynamically allocating and deallocating multi-dimensional arrays.
  - Includes functions for allocating and freeing 4D, 3D, 2D, and 1D arrays for integers and floats.
  - Ensures proper memory management and helps prevent issues like memory leaks and buffer overflows.

- [`utils/memory.c`](utils/memory.c):
  Provides implementations for the functions defined in `utils/memory.h`.

- [`utils/metric.h`](utils/metric.h):
  Provides implementations for various evaluation metrics used in machine learning.
  - Includes a function for obtaining the predicted class label from the model's output (`get_prediction_accuracy`).
  - Includes a function for computing the F1-score for a given output and true label (`compute_f1_score`).

- [`utils/metric.c`](utils/metric.c):
  Provides implementations for the functions defined in `utils/metric.h`.

- [`utils/optim.h`](utils/optim.h):
  Provides implementations for various optimization algorithms used in deep learning.
  - Includes implementations for Stochastic Gradient Descent (SGD) with momentum (`sgd`).
  - Includes an implementation for the Adam optimization algorithm (`adam`).
  - Includes an implementation for the RMSprop optimization algorithm (`rmsprop`).

- [`utils/optim.c`](utils/optim.c):
  Provides implementations for the functions defined in `utils/optim.h`.

- [`utils/rand.h`](utils/rand.h):
  Provides utility functions for generating random numbers.
  - Includes a function for generating random floating-point numbers within a specified range (`rand_uniform`).
  - Includes a function for generating random integers within a specified range (`rand_int`).

- [`utils/rand.c`](utils/rand.c):
  Provides implementations for the functions defined in `utils/rand.h`.

- [`utils/tensor.h`](utils/tensor.h):
  Provides utility functions for working with tensors in deep learning applications.
  - Includes functions for allocating memory for output tensors and gradient tensors (`allocate_output_tensor`, `allocate_grad_tensor`).
  - Includes a function for copying the contents of a tensor to a new tensor (`copy_3d_array`).
  - Includes a function for freeing the memory allocated for a tensor (`free_tensor`).

- [`utils/tensor.c`](utils/tensor.c):
  Provides implementations for the functions defined in `utils/tensor.h`.

- [`utils/tools.h`](utils/tools.h):
  Defines various utility functions.

- [`utils/tools.c`](utils/tools.c):
  Provides implementations for the utility functions defined in `utils/tools.h`.

- [`utils/train.h`](utils/train.h):
  Provides utility functions for training and evaluating deep learning models.
  - Includes functions for computing the loss for single outputs and batches of outputs (`compute_loss`, `compute_loss_batch`).
  - Includes functions for computing the gradients of the output with respect to the loss function (`compute_output_grad`, `compute_output_grad_batch`).
  - Includes a function for generating predictions from the model's output based on the specified evaluation metric (`get_prediction`).
  - Includes a function for computing the accuracy of the model's predictions for a batch of outputs (`compute_accuracy`).

- [`utils/train.c`](utils/train.c):
  Provides implementations for the functions defined in `utils/train.h`.

- [`utils/utils.h`](utils/utils.h):
  Includes all the above `.h` files of utils folder, serving as the entry point for the utility library.

- [`main.c`](main.c):
  Contains the `main` function for testing and demonstrating the usage of the CNN API.

### 中文说明

- [`CNN.h`](CNN.h):
  提供与 CNN 模型交互的高级接口函数。该头文件包括创建、编译、训练、测试和释放 CNN 模型的函数。
  - 定义诸如 `create_model` 用于初始化具有指定输入和输出维度的新 CNN 模型、`add_layer` 用于向模型添加层、`compile_model` 用于为模型分配配置设置、`train_model` 用于使用提供的训练数据训练模型、`test_model` 用于使用测试数据评估模型的函数。
  - 另外还包括 `print_model_info` 用于显示模型信息、`check_output_shape` 用于验证最后一层输出形状是否与输出信息匹配、`free_model` 用于释放为模型分配的内存的函数。

- [`CNN.c`](CNN.c):
  提供 `CNN.h` 中定义的函数的实现。

- [`dataset.h`](dataset.h):
  提供用于管理 CNN API 中使用的数据集的定义和函数。
  - 定义 `Dataset` 结构体,描述数据集,包括元数据(如名称、批次大小和图像数量)以及指向输入数据和标签的指针。
  - 提供从 JSON 文件加载数据集、从数据集创建 JSON 文件、从数据集创建批次、将数据集分割为批次以及释放为数据集分配的内存的函数。

- [`dataset.c`](dataset.c):
  提供从 JSON 文件加载数据集、从数据集创建 JSON 文件、将数据集分割为批次以及释放为数据集分配的内存的函数的实现。

- [`input/data.h`](input/data.h):
  定义用于处理输入数据(包括图像)的结构和函数,用于 `dataset.h`。
  - 定义 `Dimensions` 结构体,表示输入数据的宽度、高度和通道数。
  - 定义 `DataType` 枚举,指定输入数据的数据类型。
  - 定义 `InputData` 联合体,用于以整数或浮点数格式保存输入数据。
  - 提供从图像加载输入数据的函数(`load_input_data_from_image` 和 `load_image_data_with_format`),支持多种图像格式(JPEG、PNG 等)。
  - 提供以浮点数或整数格式从磁盘加载特定图像格式的函数(`loadFloatJPEG`、`loadIntJPEG`、`loadFloatPNG`、`loadIntPNG`、`loadFloatImage`、`loadIntImage`)。
  - 包括创建空输入数据(`create_empty_input_data`)、调整图像大小(`resize_image`)和释放与输入数据相关内存(`free_image_data`)的函数。

- [`input/input.c`](input/input.c):
  提供从图像加载输入数据、调整图像大小以及释放与输入数据相关内存的函数的实现。

- [`model/layer/layer.h`](model/layer/layer.h):
  定义 CNN 模型中层对象的接口。该头文件包括层类型的定义、层参数、权重和偏置的结构体定义，以及创建、初始化、执行前向传递、后向传递、更新参数和删除层的函数声明。
  - 定义 `LayerType` 枚举,表示不同类型的层:卷积层、池化层、全连接层和丢弃层。
  - 定义 `LayerParams` 联合体,用于保存每种层类型的特定参数。
  - 定义 `Layer` 结构体,表示 CNN 模型中的一层,包括其类型、参数、权重、偏置以及指向模型中下一层的指针。

- [`model/layer/layer.c`](model/layer/layer.c):
  提供 `model/layer/layer.h` 中定义的函数的实现，包括创建层、初始化层、执行前向传递、后向传递、更新参数、重置梯度、计算输出形状和删除层等功能。

- [`model/model.h`](model/model.h):
  提供与CNN模型交互的高级接口函数。
  - 定义了`Model`结构体，用于保存模型配置和层。
  - 包括创建新的CNN模型(`create_model`)、向模型添加层(`add_layer`、`add_layer_to_model`)、使用配置设置编译模型(`compile_model`)、执行前向和反向传播(`forward_pass`、`backward_pass`、`forward_pass_batch`、`backward_pass_batch`)的函数。
  - 在数据集上训练模型时将使用训练方法(`train_model`)，评估模型时将需要使用评估方法(`evaluate_model`)。
  - 关于前向传播，还包括更新模型权重(`update_model_weights`)和重置梯度(`reset_model_grads`)。
  - 另外，还包括打印模型信息(`print_model_info`)以显示整个模型信息包括每一层，以及释放为模型分配的内存(`delete_model`)。

- [`model/model.c`](model/model.c):
  提供`model/model.h`中定义的函数的实现。

- [`optimizer/optimizer.h`](optimizer/optimizer.h):
  定义了在训练过程中使用的不同优化算法的接口。
  - 定义了`OptimizerType` enum来表示不同的优化器类型(SGD、Adam、RMSprop)。
  - 为每种优化器类型定义了单独的结构体(`SGDOptimizer`、`AdamOptimizer`、`RMSpropOptimizer`)来保存相应的参数和状态变量。
  - 定义了`Optimizer`结构体来保存优化器类型和不同优化器结构体的联合体。
  - 提供了函数声明,用于初始化不同优化器类型(`init_sgd`、`init_adam`、`init_rmsprop`)、创建Optimizer实例(`create_optimizer`)以及释放优化器使用的内存(`delete_optimizer`)。

- [`optimizer/optimizer.c`](optimizer/optimizer.c):
  提供了`optimizer/optimizer.h`中定义的函数的实现。

- [`utils/compute/activation.h`](utils/compute/activation.h):
  提供神经网络中常用的各种激活函数的实现。
  - 定义了不同激活函数(如ReLU、Sigmoid、Tanh、Max和Softmax)的前向和反向传播函数。
  - 包含了在前向传播时应用激活函数的统一接口(`apply_activation`)。
  - 包含了在反向传播时计算梯度的统一接口(`apply_activation_backward`)。

- [`utils/compute/activation.c`](utils/compute/activation.c):
  提供`utils/compute/activation.h`中定义的函数的实现。

- [`utils/compute/convolutional.h`](utils/compute/convolutional.h):
  提供卷积神经网络(CNN)中执行卷积操作的实现。
  - 定义了卷积层的前向和反向传播函数(`conv_forward`、`conv_backward`)。
  - 定义了使用优化算法更新卷积层权重和偏置的函数(`update_conv_weights`)。

- [`utils/compute/convolutional.c`](utils/compute/convolutional.c):
  提供`utils/compute/convolutional.h`中定义的函数的实现。

- [`utils/compute/dropout.h`](utils/compute/dropout.h):
  提供神经网络中使用的dropout正则化技术的实现。
  - 定义了dropout层的前向传播函数(`dropout_forward`)。
  - 定义了dropout层的反向传播函数(`dropout_backward`)。

- [`utils/compute/dropout.c`](utils/compute/dropout.c):
  提供`utils/compute/dropout.h`中定义的函数的实现。

- [`utils/compute/flatten.h`](utils/compute/flatten.h):
  提供神经网络中使用的flatten和unflatten操作的实现。
  - 定义了将多维张量压平为一维向量的函数(`flatten`)。
  - 定义了将一维向量解压缩为多维张量的函数(`unflatten`)。

- [`utils/compute/flatten.c`](utils/compute/flatten.c):
  提供`utils/compute/flatten.h`中定义的函数的实现。

- [`utils/compute/fully_connected.h`](utils/compute/fully_connected.h):
  提供神经网络中使用的全连接层的实现。
  - 定义了全连接层的前向传播函数(`fc_forward`)。
  - 定义了全连接层的反向传播函数(`fc_backward`)。
  - 定义了使用优化算法更新全连接层权重和偏置的函数(`update_fc_weights`)。

- [`utils/compute/fully_connected.c`](utils/compute/fully_connected.c):
  提供`utils/compute/fully_connected.h`中定义的函数的实现。

- [`utils/compute/pooling.h`](utils/compute/pooling.h):
  提供卷积神经网络(CNN)中使用的池化操作的实现。
  - 定义了池化层的前向传播函数(`pool_forward`)。
  - 定义了池化层的反向传播函数(`pool_backward`)。

- [`utils/compute/pooling.c`](utils/compute/pooling.c):
  提供`utils/compute/pooling.h`中定义的函数的实现。

- [`utils/dimension.h`](utils/dimension.h):
  定义维度相关的结构体和函数。

- [`utils/loss.h`](utils/loss.h):
  提供深度学习中使用的各种损失函数的实现。
  - 定义了`LossFunction`类型为函数指针。
  - 包含了分类交叉熵损失函数和均方误差损失函数的实现。
  - 允许开发人员根据`LossFunction`签名实现和集成自定义损失函数。

- [`utils/loss.c`](utils/loss.c):
  提供`utils/loss.h`中定义的函数的实现。

- [`utils/memory.h`](utils/memory.h):
  提供动态分配和释放多维数组的实用函数。
  - 包括分配和释放4D、3D、2D和1D整数和浮点数数组的函数。
  - 确保正确的内存管理,并有助于防止内存泄漏和缓冲区溢出等问题。

- [`utils/memory.c`](utils/memory.c):
  提供`utils/memory.h`中定义的函数的实现。

- [`utils/metric.h`](utils/metric.h):
  提供机器学习中使用的各种评估指标的实现。
  - 包括从模型输出获取预测类别标签的函数(`get_prediction_accuracy`)。
  - 包括计算给定输出和真实标签的F1分数的函数(`compute_f1_score`)。

- [`utils/metric.c`](utils/metric.c):
  提供`utils/metric.h`中定义的函数的实现。

- [`utils/optim.h`](utils/optim.h):
  提供深度学习中使用的各种优化算法的实现。
  - 包括带动量的随机梯度下降(SGD)优化算法的实现(`sgd`)。
  - 包括Adam优化算法的实现(`adam`)。
  - 包括RMSprop优化算法的实现(`rmsprop`)。

- [`utils/optim.c`](utils/optim.c):
  提供`utils/optim.h`中定义的函数的实现。

- [`utils/rand.h`](utils/rand.h):
  提供生成随机数的实用函数。
  - 包括在指定范围内生成随机浮点数的函数(`rand_uniform`)。
  - 包括在指定范围内生成随机整数的函数(`rand_int`)。

- [`utils/rand.c`](utils/rand.c):
  提供`utils/rand.h`中定义的函数的实现。

- [`utils/tensor.h`](utils/tensor.h):
  提供在深度学习应用中使用张量的实用函数。
  - 包括分配输出张量和梯度张量内存的函数(`allocate_output_tensor`、`allocate_grad_tensor`)。
  - 包括将张量内容复制到新张量的函数(`copy_3d_array`)。
  - 包括释放分配给张量的内存的函数(`free_tensor`)。

- [`utils/tensor.c`](utils/tensor.c):
  提供`utils/tensor.h`中定义的函数的实现

- [`utils/tools.h`](utils/tools.h):
  提供执行常见字符串操作的实用函数。
  - 包括检查给定字符串是否为空的函数(`is_empty_string`)。
  - 包括检查给定字符串是否不为空的函数(`not_empty_string`)。

- [`utils/tools.c`](utils/tools.c):
  提供`utils/tools.h`中定义的函数的实现。

- [`utils/train.h`](utils/train.h):
  提供训练和评估深度学习模型的实用函数。
  - 包括计算单个输出和批量输出的损失值的函数(`compute_loss`、`compute_loss_batch`)。
  - 包括计算输出相对于损失函数的梯度的函数(`compute_output_grad`、`compute_output_grad_batch`)。
  - 包括根据指定的评估指标(如准确度、F1分数)从模型输出生成预测的函数(`get_prediction`)。
  - 包括计算模型对批量输出预测的准确度的函数(`compute_accuracy`)。

- [`utils/train.c`](utils/train.c):
  提供`utils/train.h`中定义的函数的实现。

- [`utils/utils.h`](utils/utils.h):
  包含`utils`文件夹中所有上述 `.h` 文件,作为工具库的入口。

- [`main.c`](main.c):
  包含 `main` 函数,用于测试和演示 CNN API 的用法。

## External Libraries (外部库)

- [libpng](https://github.com/glennrp/libpng): Library for reading and writing PNG image files.
  License: [PNG Reference Library License](http://www.libpng.org/pub/png/src/libpng-LICENSE.txt)

- [jpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo): Library for reading and writing JPEG image files. 
  License: [IJG License](https://www.ijg.org/files/README)

- [json-c](https://github.com/json-c/json-c): Library for parsing JSON data. 
  License: [MIT License](https://opensource.org/licenses/MIT)

- [hdf5](https://www.hdfgroup.org/solutions/hdf5/): Library for reading and writing HDF5 files.
  License: [HDF5 License](https://github.com/HDFGroup/hdf5?tab=License-1-ov-file#)

- [libpng](https://github.com/glennrp/libpng): 用于读取和写入 PNG 图像文件的库。
  许可证: [PNG Reference Library License](http://www.libpng.org/pub/png/src/libpng-LICENSE.txt)

- [jpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo): 用于读取和写入 JPEG 图像文件的库。
  许可证: [IJG License](https://www.ijg.org/files/README)

- [json-c](https://github.com/json-c/json-c): 用于解析 JSON 数据的库。
  许可证: [MIT License](https://opensource.org/licenses/MIT)

- [hdf5](https://www.hdfgroup.org/solutions/hdf5/): 用于读取和写入 HDF5 文件的库。
  许可证: [HDF5 License](https://github.com/HDFGroup/hdf5?tab=License-1-ov-file#)
