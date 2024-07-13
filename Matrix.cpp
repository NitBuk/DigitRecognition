// Matrix.cpp
#include "Matrix.h"

// Constructs Matrix of size 1 × 1. Inits the single element to 0.
Matrix::Matrix() {
  dims_.rows = 1;
  dims_.cols = 1;
  data_ = new float*[1];
  data_[0] = new float[1];
  data_[0][0] = INITIALIZE;
}

// Constructs Matrix of size 𝑟𝑜𝑤𝑠 × 𝑐𝑜𝑙𝑠. Inits all elements to 0.
Matrix::Matrix(int rows, int cols) : dims_({rows, cols}) {
  if(rows < 1 || cols < 1) {
    throw std::length_error(LENGTH_ERROR);
  }
  data_ = new float*[rows];
  for (int i = 0; i < rows; i++) {
      data_[i] = new float[cols];
    for (int j = 0; j < cols; j++) {
      data_[i][j] = 0;
    }
  }
}

// Constructs Matrix of size 𝑟𝑜𝑤𝑠 × 𝑐𝑜𝑙𝑠. Inits all elements to Matrix m
// elements.
Matrix::Matrix(const Matrix& m) : dims_({m.dims_.rows, m.dims_.cols}) {
  data_ = new float*[dims_.rows];
  for (int i = 0; i < dims_.rows; i++){
  data_[i] = new float[dims_.cols];
  for (int j = 0; j < dims_.cols; j++){
    data_[i][j] = m.data_[i][j];
  }
  }
}

// Destructor:
// Frees all memory allocated by this Matrix.
Matrix::~Matrix() {
    for (int i = 0; i < dims_.rows; i++)
    {
        delete[] data_[i];
    }
    delete[] data_;
}

// Methods & Functions:

// Transforms a matrix into its transpose matrix.
Matrix& Matrix::transpose(){
  if(dims_.rows < 1 || dims_.cols < 1) {
    throw std::length_error(LENGTH_ERROR);
  }
  int new_rows = dims_.cols;
  int new_cols = dims_.rows;
  float** new_data = new float*[new_rows];
  for (int i = 0; i < new_rows; ++i)
  {
    new_data[i] = new float[new_cols];
  }
  for (int i = 0; i < dims_.rows; i++)
  {
    for (int j = 0; j < dims_.cols; j++)
    {
      new_data[j][i] = data_[i][j];
    }
  }
  for (int i = 0; i < dims_.rows; i++)
  {
    delete[] data_[i];
  }
  delete[] data_;
  dims_.rows = new_rows;
  dims_.cols = new_cols;
  data_ = new_data;
  return *this;
}

// Transforms a matrix into a column vector.
Matrix& Matrix::vectorize(){
  if(dims_.rows < 1 || dims_.cols < 1) {
    throw std::length_error(LENGTH_ERROR);
  }
  int new_rows = dims_.rows * dims_.cols;
  int new_cols = 1;
  float** new_data = new float*[new_rows];
  for (int i = 0; i < new_rows; ++i)
  {
    new_data[i] = new float[new_cols];
  }
  for (int i = 0; i < dims_.rows; i++)
  {
    for (int j = 0; j < dims_.cols; j++)
    {
      new_data[i * dims_.cols + j][0] = data_[i][j];
    }
  }
  for (int i = 0; i < dims_.rows; i++)
  {
    delete[] data_[i];
  }
  delete[] data_;
  dims_.rows = new_rows;
  dims_.cols = new_cols;
  data_ = new_data;
  return *this;
}

// Prints the matrix elements.
void Matrix::plain_print() const{
  if (data_ == nullptr) {throw std::runtime_error(RUN_TIME_ERROR);}
    for (int i = 0; i < dims_.rows; i++)
    {
      if (data_[i] == nullptr) {throw std::runtime_error(RUN_TIME_ERROR);}
        for (int j = 0; j < dims_.cols; j++)
        {
          std::cout << data_[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Returns a matrix which is the element-wise multiplication
// (Hadamard product) of this matrix with another matrix m.
Matrix Matrix::dot(const Matrix& m) const {
  if(dims_.rows != m.dims_.rows || dims_.cols != m.dims_.cols) {
    throw std::length_error(LENGTH_ERROR);
  }
  Matrix result(dims_.rows, dims_.cols);
  for (int i = 0; i < dims_.rows; i++)
  {
      for (int j = 0; j < dims_.cols; j++)
      {
          result.data_[i][j] = data_[i][j] * m.data_[i][j];
      }
  }
  return result;
}

// Returns the Frobenius norm of the given matrix.
float Matrix::norm() const{
  if (data_ == nullptr) {throw std::runtime_error(RUN_TIME_ERROR);}
  float result = 0;
  for (int i = 0; i < dims_.rows; i++)
  {
    if (data_[i] == nullptr) {throw std::runtime_error(RUN_TIME_ERROR);}
    for (int j = 0; j < dims_.cols; j++)
    {
      result += pow (data_[i][j], SQUARE);
    }
  }
  return sqrt (result);
}

// Swap rows i and j in matrix m.
void swap_rows(Matrix& m, int i, int j) {
  float* tmp = m.data_[i];
  m.data_[i] = m.data_[j];
  m.data_[j] = tmp;
}

// Divide row i in matrix m by div.
void divide_row(Matrix& m, int i, float div) {
  if (fabs(div) < EPSILON) {
    throw std::runtime_error("Can't divide by zero.");
  }
  for (int j = 0; j < m.dims_.cols; j++) {
    m.data_[i][j] /= div;
    if (m.data_[i][j] == -0) { m.data_[i][j] = 0; }
  }
}

// Subtract a multiple of row j from row i in matrix m.
void row_subtract(Matrix& m, int i, int j, float multiple) {
  for (int k = 0; k < m.dims_.cols; k++) {
    m.data_[i][k] -= multiple * m.data_[j][k];
    if (m.data_[i][k] == -0) { m.data_[i][k] = 0; }
  }
}

// Returns a new Matrix that is the reduced row echelon form of the original.
Matrix Matrix::rref() const {
  if (dims_.rows == 0 || dims_.cols == 0) {
    throw std::runtime_error(RUN_TIME_ERROR);
  }
  Matrix result(*this);
  int lead = 0;
  for (int r = 0; r < result.dims_.rows; r++) {
    if (result.dims_.cols <= lead) {
      return result;
    }
    int i = r;
    while (result.data_[i][lead] == 0) {
      i++;
      if (result.dims_.rows == i) {
        i = r;
        lead++;
        if (result.dims_.cols == lead) {
          return result;
        }
      }
    }
    swap_rows(result, i, r);
    float div = result.data_[r][lead];
    divide_row(result, r, div);
    for (int j = 0; j < result.dims_.rows; j++) {
      if (j != r) {
        row_subtract(result, j, r, result.data_[j][lead]);
      }
    }
    lead++;
  }
  return result;
}


// Returns index of the largest number in the matrix.
int Matrix::argmax() const{
  if (data_ == nullptr || dims_.rows <= 0 || dims_.cols <= 0) {
    throw std::runtime_error(RUN_TIME_ERROR);
  }
  float max = data_[0][0];
  int index = 0;
  for (int i = 0; i < dims_.rows; i++)
  {
    if (data_[i] == nullptr) {throw std::runtime_error(RUN_TIME_ERROR);}
    for (int j = 0; j < dims_.cols; j++)
    {
      if(data_[i][j] > max){
        max = data_[i][j];
        index = i * dims_.cols + j;
      }
    }
  }
  return index;
}

// Returns the sum of all elements in the matrix.
float Matrix::sum() const{
  if (data_ == nullptr || dims_.rows <= 0 || dims_.cols <= 0) {
    throw std::runtime_error(RUN_TIME_ERROR);
  }
  float result = 0;
  for (int i = 0; i < dims_.rows; i++)
  {
    if (data_[i] == nullptr) {throw std::runtime_error(RUN_TIME_ERROR);}
    for (int j = 0; j < dims_.cols; j++)
    {
      result += data_[i][j];
    }
  }
  return result;
}

// Operators:

// Matrix addition.
Matrix Matrix::operator+(const Matrix& m) const{
  if(dims_.rows != m.dims_.rows || dims_.cols != m.dims_.cols) {
    throw std::length_error(LENGTH_ERROR);
  }
  Matrix result(dims_.rows, dims_.cols);
  for (int i = 0; i < dims_.rows; i++)
  {
      for (int j = 0; j < dims_.cols; j++)
      {
      result.data_[i][j] = data_[i][j] + m.data_[i][j];
      }
  }
  return result;
}

// Matrix assignment.
Matrix& Matrix::operator=(const Matrix& m)
{
  if (this != &m)
  {
    for (int i = 0; i < dims_.rows; ++i)
    {
      delete[] data_[i];
    }
    delete[] data_;
    dims_ = m.dims_;
    data_ = new float*[dims_.rows];
    for (int i = 0; i < dims_.rows; ++i)
    {
      data_[i] = new float[dims_.cols];
      for (int j = 0; j < dims_.cols; ++j)
      {
        data_[i][j] = m.data_[i][j];
      }
    }
  }
  return *this;
}

// Matrix addition and assignment.
Matrix& Matrix::operator+=(const Matrix& m){
  if(dims_.rows != m.dims_.rows || dims_.cols != m.dims_.cols) {
    throw std::length_error(LENGTH_ERROR);
  }
  for (int i = 0; i < dims_.rows; i++)
  {
      for (int j = 0; j < dims_.cols; j++)
      {
      data_[i][j] += m.data_[i][j];
      }
  }
  return *this;
}

// Matrix multiplication.
Matrix Matrix::operator*(const Matrix& m) const{
  if(dims_.cols != m.dims_.rows) {throw std::length_error(LENGTH_ERROR);}
  Matrix result(dims_.rows, m.dims_.cols);
  for (int i = 0; i < dims_.rows; i++)
  {
      for (int j = 0; j < m.dims_.cols; j++)
      {
          for (int k = 0; k < dims_.cols; k++)
          {
          result.data_[i][j] += data_[i][k] * m.data_[k][j];
          }
      }
  }
  return result;
}

// Scalar multiplication on the right.
Matrix Matrix::operator*(float scalar) const{
  Matrix result(dims_.rows, dims_.cols);
  for (int i = 0; i < dims_.rows; i++)
  {
      for (int j = 0; j < dims_.cols; j++)
      {
      result.data_[i][j] = data_[i][j] * scalar;
      }
  }
  return result;
}

// Scalar multiplication on the left.
Matrix operator*(float scalar, const Matrix& m){
  Matrix result(m.dims_.rows, m.dims_.cols);
  for (int i = 0; i < m.dims_.rows; i++)
  {
      for (int j = 0; j < m.dims_.cols; j++)
      {
      result.data_[i][j] = m.data_[i][j] * scalar;
      }
  }
  return result;
}

// Parenthesis indexing.
float& Matrix::operator()(int i, int j){
  if(i < 0 || i >= dims_.rows || j < 0 || j >= dims_.cols) {
    throw std::out_of_range(OUT_OF_RANGE_ERROR);
  }
  return data_[i][j];
}
// Const parenthesis indexing.
float Matrix::operator()(int i, int j) const{
  if(i < 0 || i >= dims_.rows || j < 0 || j >= dims_.cols) {
    throw std::out_of_range(OUT_OF_RANGE_ERROR);
  }
  return data_[i][j];
}

// Brackets indexing.
float& Matrix::operator[](int i){
  if (i < 0 || i >= dims_.rows * dims_.cols) {
    throw std::out_of_range(OUT_OF_RANGE_ERROR);
  }
  return data_[i / dims_.cols][i % dims_.cols];
}

// Const brackets indexing.
float Matrix::operator[](int i) const{
  if (i < 0 || i >= dims_.rows * dims_.cols) {
    throw std::out_of_range(OUT_OF_RANGE_ERROR);
  }
  return data_[i / dims_.cols][i % dims_.cols];
}

// Output stream overload.
std::ostream& operator<<(std::ostream& os, const Matrix& m){
  for (int i = 0; i < m.dims_.rows; i++)
  {
    for (int j = 0; j < m.dims_.cols; j++)
    {
      if(m.data_[i][j] > MATRIX_PRINT_THRESHOLD){
        os << "**";
      }else{
        os << "  ";
      }
      if (!os) {
        throw std::runtime_error(RUN_TIME_ERROR);
      }
    }
    os << std::endl;
    if (!os) {
      throw std::runtime_error(RUN_TIME_ERROR);
    }
  }
  return os;
}

// Input stream overload.
std::istream& operator>>(std::istream& is, Matrix& m){
  if(!is || is.eof() || !m.data_){throw std::runtime_error(RUN_TIME_ERROR);}
  for (int i = 0; i < m.dims_.rows; i++)
  {
    for (int j = 0; j < m.dims_.cols; j++)
    {
      if (!is.read(reinterpret_cast<char*>(&m.data_[i][j]), sizeof(float)))
      {
        throw std::runtime_error(RUN_TIME_ERROR);
      }
    }
  }
  return is;
}
