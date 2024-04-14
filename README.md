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

To correctly utilize this API, please follow the instructions below:

1. Define your neural network model using the provided layers and architectures. For detailed function documentation, refer to [`CNN.h`](CNN.h).

2. For instructions on creating datasets and the standard format for datasets, please refer to [`dataset.h`](dataset.h).

For detailed usage instructions and function documentation, please refer to:

- [CNN.h](CNN.h) (Model Construction)
- [dataset.h](dataset.h) (Dataset Creation)
- [loss/losses.h](loss/losses.h) (Loss Functions)
- [optimizer/optimizer.h](optimizer/optimizer.h) (Optimizers).

要正确使用该 API，请按照以下说明进行操作：
1. 使用提供的层和架构定义您的神经网络模型。有关详细函数文档，请参阅 [`CNN.h`](CNN.h)。
2. 如何进行数据集的建立以及数据集的标准格式，请参阅 [`dataset.h`](dataset.h)。

有关详细的使用说明和函数文档，请参阅 [`CNN.h`](CNN.h)（模型构建）、[`dataset.h`](dataset.h)（数据集的建立）、[`loss/losses.h`](loss/losses.h)（损失函数）、[`optimizer/optimizer.h`](optimizer/optimizer.h)（优化器）。

## API文件 (API File)

- [CNN.h](CNN.h): Header file containing function declarations and documentation for the CNN API.
- [dataset.h](input/dataset.h): Header file containing function declarations and documentation for dataset creation and management.
  - [input/test_pic](input/test_pic): Include a demo dataset format and standard .json file [input/test_pic/dataset.json](input/test_pic/dataset.json).

## 代码文件 (Code Files)

- [CNN.h](CNN.h): 
  Provides high-level interface functions for interacting with the CNN model. This header file includes functions for creating, compiling, training, testing, and freeing the CNN model.
  - Defines functions such as `create_model` to initialize a new CNN model with specified input and output dimensions, `add_layer` to append layers to the model, `compile_model` to assign configuration settings to the model, `train_model` to train the model with provided training data, and `test_model` to evaluate the model with test data.
  - Additionally, includes functions like `print_model_info` to display information about the model, `check_output_shape` to verify if the final layer output shape matches the output information, and `free_model` to release memory allocated for the model.

- [CNN.c](CNN.c): 
  Provides implementations for the functions defined in `CNN.h`.

- [dataset.h](dataset.h): 
  Provides definitions and functions for managing datasets used in the CNN API.
  - Defines a structure `Dataset` to describe datasets, including metadata such as name, batch size, and number of images, along with pointers to input data and labels.
  - Offers functions for loading datasets from JSON files, creating JSON files from datasets, creating batches from datasets, splitting datasets into batches, and freeing memory allocated for datasets.

- [dataset.c](dataset.c): 
  Provides implementations for functions to load datasets from JSON files, create JSON files from datasets, split datasets into batches, and free memory allocated for datasets.

- [input/input.h](input/input.h): 
  Defines structures and functions for handling input data, including images, for `dataset.h`.
  - Defines a `Dimensions` structure to represent the width, height, and number of channels of input data.
  - Defines a `DataType` enum to specify the data type of input data.
  - Defines a `InputData` union to hold input data in either int or float format.
  - Provides functions for loading input data from images (`load_input_data_from_image` and `load_image_data_with_format`) with support for various image formats (JPEG, PNG, etc.).
  - Provides functions (`loadFloatJPEG`, `loadIntJPEG`, `loadFloatPNG`, `loadIntPNG`, `loadFloatImage`, `loadIntImage`) for loading images from disk in either float or int format for specific image formats.
  - Includes functions to create empty input data (`create_empty_input_data`), resize images (`resize_image`), and free memory associated with input data (`free_image_data`).

- [input/input.c](input/input.c): 
  Provides implementations for functions to load input data from images, resize images, and free memory associated with input data.

- [model/layer/layer.h](model/layer/layer.h): 
  Defines the interface for the layer object in the CNN model. The layer object contains information necessary for performing the forward pass, backward pass, and parameter update for a specific layer in the model.
  - Defines an enum `LayerType` for different types of layers: convolutional, pooling, fully connected, and dropout.
  - Defines a union `LayerParams` to hold parameters specific to each layer type.
  - Defines a struct `Layer` representing a layer in the CNN model, including its type, parameters, weights, biases, and a pointer to the next layer in the model.

- [model/model.h](model/model.h): 
  Defines the interface for the model object in the CNN. The model object contains methods to perform the forward pass, backward pass, and parameter update for the entire model.
  - Defines a struct `ModelConfig` to hold configuration settings for the model, including optimizer, learning rate, loss function, and evaluation metric.
  - Defines a struct `Model` representing the entire model, including input and output dimensions, optimizer, learning rate, loss function, evaluation metric, and a pointer to the first layer in the model.
  - Provides function declarations for performing the forward pass (`forward_pass`), backward pass (`backward_pass`), training a single epoch (`train_epoch`), and evaluating the model (`evaluate_model`).

- [model/model.c](model/model.c): 
  

## External Libraries (外部库)

- [libpng](https://github.com/glennrp/libpng): Library for reading and writing PNG image files.
  License: [PNG Reference Library License](http://www.libpng.org/pub/png/src/libpng-LICENSE.txt)

- [jpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo): Library for reading and writing JPEG image files. 
  License: [IJG License](https://www.ijg.org/files/README)

- [json-c](https://github.com/json-c/json-c): Library for parsing JSON data. 
  License: [MIT License](https://opensource.org/licenses/MIT)
