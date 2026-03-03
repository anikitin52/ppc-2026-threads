#include "morozova_s_strassen_multiplication/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace morozova_s_strassen_multiplication {

MorozovaSStrassenMultiplicationSEQ::MorozovaSStrassenMultiplicationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool MorozovaSStrassenMultiplicationSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int n = static_cast<int>(GetInput()[0]);
  size_t expected_size = 1 + 2 * static_cast<size_t>(n) * n;

  return GetInput().size() == expected_size && n > 0;
}

bool MorozovaSStrassenMultiplicationSEQ::PreProcessingImpl() {
  n_ = static_cast<int>(GetInput()[0]);

  A_ = Matrix(n_);
  B_ = Matrix(n_);

  int idx = 1;
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      A_(i, j) = GetInput()[idx++];
    }
  }

  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      B_(i, j) = GetInput()[idx++];
    }
  }

  return true;
}

bool MorozovaSStrassenMultiplicationSEQ::RunImpl() {
  int leaf_size = 64;

  if (n_ <= leaf_size) {
    C_ = MultiplyStandard(A_, B_);
  } else {
    C_ = MultiplyStrassen(A_, B_, leaf_size);
  }

  return true;
}

bool MorozovaSStrassenMultiplicationSEQ::PostProcessingImpl() {
  OutType &output = GetOutput();
  output.clear();

  output.push_back(static_cast<double>(n_));

  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      output.push_back(C_(i, j));
    }
  }

  return true;
}

Matrix MorozovaSStrassenMultiplicationSEQ::AddMatrix(const Matrix &A, const Matrix &B) const {
  int n = A.size;
  Matrix result(n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result(i, j) = A(i, j) + B(i, j);
    }
  }

  return result;
}

Matrix MorozovaSStrassenMultiplicationSEQ::SubtractMatrix(const Matrix &A, const Matrix &B) const {
  int n = A.size;
  Matrix result(n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result(i, j) = A(i, j) - B(i, j);
    }
  }

  return result;
}

Matrix MorozovaSStrassenMultiplicationSEQ::MultiplyStandard(const Matrix &A, const Matrix &B) const {
  int n = A.size;
  Matrix result(n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

  return result;
}

void MorozovaSStrassenMultiplicationSEQ::SplitMatrix(const Matrix &M, Matrix &M11, Matrix &M12, Matrix &M21,
                                                     Matrix &M22) const {
  int n = M.size;
  int half = n / 2;

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      M11(i, j) = M(i, j);
      M12(i, j) = M(i, j + half);
      M21(i, j) = M(i + half, j);
      M22(i, j) = M(i + half, j + half);
    }
  }
}

Matrix MorozovaSStrassenMultiplicationSEQ::MergeMatrices(const Matrix &M11, const Matrix &M12, const Matrix &M21,
                                                         const Matrix &M22) const {
  int half = M11.size;
  int n = 2 * half;
  Matrix result(n);

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      result(i, j) = M11(i, j);
      result(i, j + half) = M12(i, j);
      result(i + half, j) = M21(i, j);
      result(i + half, j + half) = M22(i, j);
    }
  }

  return result;
}

Matrix MorozovaSStrassenMultiplicationSEQ::MultiplyStrassen(const Matrix &A, const Matrix &B, int leaf_size) const {
  int n = A.size;

  if (n <= leaf_size) {
    return MultiplyStandard(A, B);
  }

  if (n % 2 != 0) {
    return MultiplyStandard(A, B);
  }

  int half = n / 2;

  Matrix A11(half), A12(half), A21(half), A22(half);
  Matrix B11(half), B12(half), B21(half), B22(half);

  SplitMatrix(A, A11, A12, A21, A22);
  SplitMatrix(B, B11, B12, B21, B22);

  Matrix P1 = MultiplyStrassen(A11, SubtractMatrix(B12, B22), leaf_size);
  Matrix P2 = MultiplyStrassen(AddMatrix(A11, A12), B22, leaf_size);
  Matrix P3 = MultiplyStrassen(AddMatrix(A21, A22), B11, leaf_size);
  Matrix P4 = MultiplyStrassen(A22, SubtractMatrix(B21, B11), leaf_size);
  Matrix P5 = MultiplyStrassen(AddMatrix(A11, A22), AddMatrix(B11, B22), leaf_size);
  Matrix P6 = MultiplyStrassen(SubtractMatrix(A12, A22), AddMatrix(B21, B22), leaf_size);
  Matrix P7 = MultiplyStrassen(SubtractMatrix(A11, A21), AddMatrix(B11, B12), leaf_size);

  Matrix C11 = AddMatrix(SubtractMatrix(AddMatrix(P5, P4), P2), P6);
  Matrix C12 = AddMatrix(P1, P2);
  Matrix C21 = AddMatrix(P3, P4);
  Matrix C22 = SubtractMatrix(SubtractMatrix(AddMatrix(P5, P1), P3), P7);

  return MergeMatrices(C11, C12, C21, C22);
}

}  // namespace morozova_s_strassen_multiplication
