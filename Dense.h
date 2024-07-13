#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"
#include "Matrix.h"

// A class representing a Dense layer.
class Dense {
 private:
  Matrix weights_;
  Matrix bias_;
  ActivationFunction activation_;

 public:
  // Constructor:
  Dense(Matrix weights, Matrix bias, ActivationFunction activation);

  // Methods

  // Returns the weights of this layer.
  Matrix get_weights() const { return weights_; }

  // Returns the bias of this layer.
  Matrix get_bias() const { return bias_; }

  // Returns the activation function of this layer.
  ActivationFunction get_activation() const { return activation_; }

  // Operators:

  // Applies the layer on input and returns output matrix.
  Matrix operator()(const Matrix& input) const;
};


#endif //DENSE_H
