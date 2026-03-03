#pragma once

#include "morozova_s_strassen_multiplication/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozova_s_strassen_multiplication {

class MorozovaSStrassenMultiplicationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MorozovaSStrassenMultiplicationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  Matrix AddMatrix(const Matrix &A, const Matrix &B) const;
  Matrix SubtractMatrix(const Matrix &A, const Matrix &B) const;
  Matrix MultiplyStrassen(const Matrix &A, const Matrix &B, int leaf_size = 64) const;
  Matrix MultiplyStandard(const Matrix &A, const Matrix &B) const;
  void SplitMatrix(const Matrix &M, Matrix &M11, Matrix &M12, Matrix &M21, Matrix &M22) const;
  Matrix MergeMatrices(const Matrix &M11, const Matrix &M12, const Matrix &M21, const Matrix &M22) const;

  Matrix A_, B_, C_;
  int n_;
};
}  // namespace morozova_s_strassen_multiplication
