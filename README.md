# Handwritten Digit Recognition using Neural Network

## Overview
This project implements a neural network in C++ for recognizing handwritten digits. The neural network is trained to identify digits from 0 to 9 with an accuracy of approximately 96%. The project demonstrates the use of various C++ features such as classes, operator overloading, and exception handling.

## Features
- **Matrix Operations**: Implements matrix creation, transposition, and various matrix manipulations.
- **Activation Functions**: Includes ReLU and Softmax activation functions.
- **Layer Management**: Supports creating and managing dense layers in the neural network.
- **Digit Prediction**: Predicts the digit in an input image using the trained neural network.

## Getting Started

### Prerequisites
- C++ compiler (e.g., g++)
- Makefile for building the project

### Building the Project
To build the project, navigate to the project directory and run:
```sh
make mlpnetwork
```

### Running the Program
The program expects binary files for weights and biases as input. To run the program, use the following command:
```sh
./mlpnetwork w1 w2 w3 w4 b1 b2 b3 b4
```
Once running, the program will prompt for the path to an image file containing a digit.

### Example Usage
After providing the paths for weights and biases, the program will wait for an image file path. Upon providing the image file, the program will:
1. Load the image into a matrix.
2. Pass the matrix through the neural network.
3. Output the predicted digit and its probability.

## Code Structure
- **Matrix.h / Matrix.cpp**: Implements the Matrix class with necessary operations.
- **Activation.h / Activation.cpp**: Implements activation functions (ReLU and Softmax).
- **Dense.h / Dense.cpp**: Implements a dense layer in the neural network.
- **MlpNetwork.h / MlpNetwork.cpp**: Implements the multi-layer perceptron network.

## Example
```sh
./mlpnetwork path/to/weights/w1 path/to/weights/w2 path/to/weights/w3 path/to/weights/w4 path/to/biases/b1 path/to/biases/b2 path/to/biases/b3 path/to/biases/b4
```
Input an image file path when prompted:
```sh
path/to/image.png
```
The program will display the predicted digit and its probability.

## Acknowledgements

Feel free to reach out with any questions or suggestions. Enjoy exploring the neural network implementation!
