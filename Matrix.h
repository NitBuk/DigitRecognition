// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>
#define EPSILON 1e-10
#define MATRIX_PRINT_THRESHOLD  0.1f
#define INITIALIZE 0.0f
#define LENGTH_ERROR "Invalid dimensions."
#define OUT_OF_RANGE_ERROR "index is out of range."
#define RUN_TIME_ERROR "Wrong arguments or input."
#define SQUARE 2
/**
 * @struct matrix_dims
 * @brief Matrix dimensions container. Used in MlpNetwork.h and main.cpp
 */

// a struct that holds the dimensions of a matrix.
typedef struct matrix_dims
{
	int rows, cols;
} matrix_dims;

// a class that represents a matrix.
class Matrix {
 private:
  float** data_;
  matrix_dims dims_;

 public:
  // Constructors:

  // Constructs Matrix of size 1 × 1. Inits the single element to 0.
  Matrix();

  // Constructs Matrix of size 𝑟𝑜𝑤𝑠 × 𝑐𝑜𝑙𝑠. Inits all elements to 0.
  Matrix(int rows, int cols);

  // Constructs a matrix from another Matrix m.
  Matrix(const Matrix& m);

  // Destructor:
  // Frees all memory allocated by this Matrix.
  ~Matrix();

  // Methods & Functions:

  // returns the amount of rows as int.
  int get_rows() const{return dims_.rows;}

  // returns the amount of columns as int.
  int get_cols() const{return dims_.cols;}

// Transforms a matrix into its transpose matrix.
  Matrix& transpose();

// Transforms a matrix into a column vector.
  Matrix& vectorize();

// Prints the matrix elements.
  void plain_print() const;

// Returns a matrix which is the element-wise multiplication
// (Hadamard product) of this matrix with another matrix m.
  Matrix dot(const Matrix& m) const;

// Returns the Frobenius norm of the given matrix.
  float norm() const;

// Swap rows i and j in matrix m.
  friend void swap_rows(Matrix& m, int i, int j);

// Divide row i in matrix m by div.
  friend void divide_row(Matrix& m, int i, float div);

// Subtract a multiple of row j from row i in matrix m.
  friend void row_subtract(Matrix& m, int i, int j, float multiple);

// Returns a new Matrix that is the reduced row echelon form of the original.
// Note: This function is a bonus, and is not required.
  Matrix rref() const;

// Returns index of the largest number in the matrix.
  int argmax() const;

// Returns the sum of all elements in the matrix.
  float sum() const;

  // Operators:

// Matrix addition.
  Matrix operator+(const Matrix& m) const;

// Matrix assignment.
  Matrix& operator=(const Matrix& m);

// Matrix addition and assignment.
  Matrix& operator+=(const Matrix& m);

// Matrix multiplication.
  Matrix operator*(const Matrix& m) const;

// Scalar multiplication on the right.
  Matrix operator*(float scalar) const;

// Scalar multiplication on the left.
  friend Matrix operator*(float scalar, const Matrix& m);

// Parenthesis indexing.
  float& operator()(int i, int j);

// Const parenthesis indexing.
  float operator()(int i, int j) const;

// Brackets indexing.
  float& operator[](int i);

// Const brackets indexing.
  float operator[](int i) const;

// Output stream overload.
  friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

// Input stream overload.
  friend std::istream& operator>>(std::istream& is, Matrix& m);
};

#endif //MATRIX_H