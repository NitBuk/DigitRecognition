// Dense.cpp
#include "Dense.h"

// Constructs a Dense layer with the given weights, bias, and activation
Dense::Dense(Matrix weights, Matrix bias, ActivationFunction activation)
    : weights_(weights), bias_(bias), activation_(activation) {
  if(weights_.get_rows() != bias_.get_rows() || bias.get_cols() != 1) {
    throw std::length_error(LENGTH_ERROR);
  }
}

// Returns the weights of the layer.
Matrix Dense::operator()(const Matrix& input) const {
  if (input.get_cols() != weights_.get_rows()) {
    throw std::out_of_range(LENGTH_ERROR);
  }
  if (bias_.get_rows() != weights_.get_rows()) {
    throw std::out_of_range(LENGTH_ERROR);
  }

  Matrix result(input.get_rows(), weights_.get_cols());
  try {
    result = input * weights_;
    result += bias_;
    result = activation_(result);
  } catch (const std::exception& e) {
    throw std::runtime_error(RUN_TIME_ERROR);
  }
  return result;
}
