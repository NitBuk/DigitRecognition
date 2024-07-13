// MlpNetwork.cpp
#include "MlpNetwork.h"
#define W_0_ROWS 128
#define W_0_COLS 784
#define W_1_ROWS 64
#define W_1_COLS 128
#define W_2_ROWS 20
#define W_2_COLS 64
#define W_3_ROWS 10
#define W_3_COLS 20

// Constructs a MlpNetwork with the given layers.
MlpNetwork::MlpNetwork(Matrix weights[MLP_SIZE], Matrix bias[MLP_SIZE])
    : layers_{Dense(weights[0], bias[0], activation::relu),
              Dense(weights[1], bias[1], activation::relu),
              Dense(weights[2], bias[2], activation::relu),
              Dense(weights[3], bias[3], activation::softmax)} {
  if (layers_[0].get_weights().get_rows() != W_0_ROWS ||
      layers_[0].get_weights().get_cols() != W_0_COLS ||
      layers_[1].get_weights().get_rows() != W_1_ROWS ||
      layers_[1].get_weights().get_cols() != W_1_COLS ||
      layers_[2].get_weights().get_rows() != W_2_ROWS ||
      layers_[2].get_weights().get_cols() != W_2_COLS ||
      layers_[3].get_weights().get_rows() != W_3_ROWS ||
      layers_[3].get_weights().get_cols() != W_3_COLS) {
    throw std::length_error(LENGTH_ERROR);
  }
}
// Returns the result of the network on the given input.
digit MlpNetwork::operator()( Matrix& input) const {
  Matrix result;
  try {
    result = input.vectorize();
    for (int i = 0; i < MLP_SIZE; i++) {
      Matrix temp = layers_[i].get_weights()*result + layers_[i].get_bias();
      result = layers_[i].get_activation()(temp);
    }
  } catch (const std::length_error& le) {
    throw std::length_error(LENGTH_ERROR);
  } catch (const std::out_of_range& oor) {
    throw std::out_of_range(OUT_OF_RANGE_ERROR);
  } catch (const std::exception& e) {
    throw std::runtime_error(RUN_TIME_ERROR);
  }
  digit d;
  try {
    d.value = result.argmax();
    d.probability = result(d.value, 0);
  } catch (const std::out_of_range& oor) {
    throw std::out_of_range(OUT_OF_RANGE_ERROR);
  } catch (const std::exception& e) {
    throw std::runtime_error(RUN_TIME_ERROR);
  }
  return d;
}
