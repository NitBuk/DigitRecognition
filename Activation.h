#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "Matrix.h"
#include "cmath"
typedef Matrix (*ActivationFunction)(const Matrix&);

// a namespace for activation functions.
namespace activation {
  // Returns a matrix which is the element-wise application of Relu on
  // matrix m.
  Matrix relu(const Matrix& m);

  // Returns a matrix which is the element-wise application of Softmax on
  // matrix m.
  Matrix softmax(const Matrix& m);
}

#endif //ACTIVATION_H