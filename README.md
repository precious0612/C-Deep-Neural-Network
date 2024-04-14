# Convolutional Neural Network (CNN) API

## 项目简介 (Project Introduction)

- The aim of this design is to create an open-source machine learning algorithm library encapsulated in the C language. It will offer a comprehensive range of algorithms and interfaces tailored to the diverse needs of machine learning development across various contexts. The significance of this endeavor is multifold:

  - Facilitating machine learning developers with a convenient, efficient, and extensible codebase in C. This empowers them to directly implement a wide array of machine learning tasks, spanning classification, regression, clustering, recommendation systems, image processing, natural language processing, and more. By eliminating the need for additional languages or frameworks, developers can streamline their workflows.

  - Providing machine learning developers with an adaptable, compatible, and high-performance codebase suitable for deployment across diverse devices. This ensures that developers can seamlessly develop and deploy machine learning applications using C across a spectrum of hardware platforms and operating systems. Whether it's embedded systems, IoT devices, mobile platforms, or cloud servers, the library enables consistent performance and functionality.

  - Offering a concise, lucid, and accessible codebase for machine learning education and research. By providing clear and easily understandable code, learners and researchers can delve into the principles, methodologies, technologies, and applications of machine learning. Additionally, they can gain insights into the advantages and limitations of employing the C language within the realm of machine learning, fostering a deeper understanding and mastery of the field.

中文：
- 该设计的目的是开源出一个基于C语言所封装的机器学习算法库，提供各类算法与接口，以满足不同场景和需求的机器学习开发。此次设计的意义在于：  
  - 为C语言的机器学习开发者提供一个方便，高效，可扩展的代码库，使得他们可以在不依赖其他语言或框架的情况下，使用C语言直接实现各种机器学习任务，如分类，回归，聚类，推荐，图像处理，自然语言处理等。 
  - 为各种设备的机器学习开发者提供一个适应性强，兼容性好，性能优的代码库，使得他们可以在不同的硬件平台和操作系统上，使用C语言开发和部署机器学习应用，如嵌入式系统，物联网设备，移动设备，云端服务器等。 
  - 为机器学习的教育和研究提供一个简洁，清晰，易懂的代码库，使得学习者和研究者可以通过C语言，深入理解和掌握机器学习的原理，方法，技术和应用，以及C语言在机器学习领域的优势和局限。

## API 使用方法 (API Usage)

To correctly utilize this API, please follow the instructions below:

1. Define your neural network model using the provided layers and architectures. For detailed function documentation, refer to [`CNN.h`](CNN.h).

2. For instructions on creating datasets and the standard format for datasets, please refer to [`input/dataset.h`](input/dataset.h).

For detailed usage instructions and function documentation, please refer to:

- [`CNN.h`](CNN.h) (Model Construction)
- [`input/dataset.h`](input/dataset.h) (Dataset Creation)
- [`loss/losses.h`](loss/losses.h) (Loss Functions)
- [`optimizer/optimizer.h`](optimizer/optimizer.h) (Optimizers).

要正确使用该 API，请按照以下说明进行操作：
1. 使用提供的层和架构定义您的神经网络模型。有关详细函数文档，请参阅 [`CNN.h`](CNN.h)。
2. 如何进行数据集的建立以及数据集的标准格式，请参阅 [`input/dataset.h`](input/dataset.h)。

有关详细的使用说明和函数文档，请参阅 [`CNN.h`](CNN.h)（模型构建）、[`input/dataset.h`](input/dataset.h)（数据集的建立）、[`loss/losses.h`](loss/losses.h)（损失函数）、[`optimizer/optimizer.h`](optimizer/optimizer.h)（优化器）。

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

