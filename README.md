# Convolutional Neural Network (CNN) API

## 项目简介 (Project Introduction)

English:
The Convolutional Neural Network (CNN) API is a deep learning library designed to facilitate the creation, training, and evaluation of CNN models. It provides a user-friendly interface for building custom CNN architectures, training them on image datasets, and evaluating their performance.

Chinese:
卷积神经网络 (CNN) API 是一个深度学习库，旨在简化 CNN 模型的创建、训练和评估。它提供了一个用户友好的接口，用于构建自定义的 CNN 结构，对图像数据集进行训练，并评估其性能。

## API 使用方法 (API Usage)

English:
To use the CNN API, follow these steps:
1. Define your CNN model using the provided layers and architecture. Refer to `CNN.h` for detailed function documentation.
2. Prepare your input data using the data representation module.
3. Train your model using the training functionality.
4. Evaluate your model using the testing and evaluation functions.

For detailed usage instructions and function documentation, refer to [`CNN.h`](CNN.h).

Chinese:
要使用 CNN API，请按照以下步骤进行操作：
1. 使用提供的层和架构定义您的 CNN 模型。有关详细函数文档，请参阅 `CNN.h`。
2. 使用数据表示模块准备输入数据。
3. 使用训练功能训练您的模型。
4. 使用测试和评估功能评估您的模型。

有关详细的使用说明和函数文档，请参阅 [`CNN.h`](CNN.h)。

## 文件链接 (File Links)

- [CNN.h](CNN.h): Header file containing function declarations and documentation for the CNN API.

## 代码文件 (Code Files)

1. [convolutional_layer.h](code/convolutional_layer.h): Header file defining the convolutional layer structure and functions.
2. [pooling_layer.h](code/pooling_layer.h): Header file defining the pooling layer structure and functions.
3. [fully_connected_layer.h](code/fully_connected_layer.h): Header file defining the fully connected layer structure and functions.
4. [model_structure.h](code/model_structure.h): Header file defining the overall CNN model structure.
5. [forward_pass.c](code/forward_pass.c): Implementation of the forward pass function.
6. [backward_pass.c](code/backward_pass.c): Implementation of the backward pass function.
7. [loss_functions.c](code/loss_functions.c): Implementation of various loss functions.
8. [optimizer.c](code/optimizer.c): Implementation of optimization algorithms.
9. [training_functionality.c](code/training_functionality.c): Implementation of training functions.
10. [testing_evaluation.c](code/testing_evaluation.c): Implementation of testing and evaluation functions.
11. [data_representation.h](code/data_representation.h): Header file defining input data representation structures and functions.

## External Libraries (外部库)

- [libpng](https://github.com/glennrp/libpng): Library for reading and writing PNG image files.
  - License: [PNG Reference Library License](https://opensource.org/licenses/libpng)

