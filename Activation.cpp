#include "Activation.h"

// Returns a matrix which is the element-wise application of Relu on matrix m.
Matrix activation::relu(const Matrix& m) {
  Matrix result(m.get_rows(), m.get_cols());
  for (int i = 0; i < m.get_rows(); i++) {
    for (int j = 0; j < m.get_cols(); j++) {
      if (std::isnan(m(i, j)) || std::isinf(m(i, j))) {
        throw std::runtime_error(RUN_TIME_ERROR);
      }
      result(i, j) = std::max(INITIALIZE, m(i, j));
    }
  }
  return result;
}


// Returns a matrix which is the element-wise application of Softmax.
Matrix activation::softmax(const Matrix& m) {
  float sum = 0;
  Matrix result(m.get_rows(), m.get_cols());
  for (int i = 0; i < m.get_rows(); i++) {
    for (int j = 0; j < m.get_cols(); j++) {
      float temp = exp(m(i, j));
      if (std::isinf(temp)) {
        throw std::runtime_error(RUN_TIME_ERROR);
      }
      sum += temp;
    }
  }
  if (sum == 0) {
    throw std::runtime_error(RUN_TIME_ERROR);
  }
  for (int i = 0; i < m.get_rows(); i++) {
    for (int j = 0; j < m.get_cols(); j++) {
      result(i, j) = exp(m(i, j)) / sum;
    }
  }
  return result;
}
