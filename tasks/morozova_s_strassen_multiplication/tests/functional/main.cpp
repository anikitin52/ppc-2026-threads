#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "morozova_s_strassen_multiplication/common/include/common.hpp"
#include "morozova_s_strassen_multiplication/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace morozova_s_strassen_multiplication {

class MorozovaSStrassenMultiplicationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_number = std::get<0>(params);

    switch (test_number) {
      case 1: {
        input_data_ = {2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        break;
      }
      case 2: {
        input_data_ = {4.0};
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            input_data_.push_back(i == j ? 1.0 : 0.0);
          }
        }
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            input_data_.push_back(i * 4 + j + 1.0);
          }
        }
        break;
      }
      case 3: {
        input_data_ = {8.0};
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 8; ++j) {
            input_data_.push_back(static_cast<double>(i + 1) * (j + 1) * 0.5);
          }
        }
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 8; ++j) {
            input_data_.push_back(static_cast<double>(i + j + 1) * 0.3);
          }
        }
        break;
      }
      case 4: {
        input_data_ = {16.0};
        for (int i = 0; i < 16; ++i) {
          for (int j = 0; j < 16; ++j) {
            input_data_.push_back(std::sin(static_cast<double>(i + j)) * 10.0);
          }
        }
        for (int i = 0; i < 16; ++i) {
          for (int j = 0; j < 16; ++j) {
            input_data_.push_back(std::cos(static_cast<double>(i - j)) * 5.0);
          }
        }
        break;
      }
      case 5: {
        input_data_ = {32.0};
        for (int i = 0; i < 32; ++i) {
          for (int j = 0; j < 32; ++j) {
            input_data_.push_back(static_cast<double>(i * 32 + j + 1));
          }
        }
        for (int i = 0; i < 32; ++i) {
          for (int j = 0; j < 32; ++j) {
            input_data_.push_back(static_cast<double>((i + j) * 2 + 1));
          }
        }
        break;
      }
      case 6: {
        input_data_ = {};
        break;
      }
      case 7: {
        input_data_ = {0.0, 1.0, 2.0, 3.0, 4.0};
        break;
      }
      default:
        input_data_ = {2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0};
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_number = std::get<0>(params);
    if (test_number == 6 || test_number == 7) {
      return true;
    }

    int n = static_cast<int>(input_data_[0]);
    Matrix A(n), B(n), expected(n);

    int idx = 1;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        A(i, j) = input_data_[idx++];
      }
    }

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        B(i, j) = input_data_[idx++];
      }
    }

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
          sum += A(i, k) * B(k, j);
        }
        expected(i, j) = sum;
      }
    }

    if (output_data.empty() || static_cast<int>(output_data[0]) != n) {
      return false;
    }

    const double eps = 1e-6;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        double result_val = output_data[1 + i * n + j];
        double expected_val = expected(i, j);
        if (std::abs(result_val - expected_val) > eps) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(MorozovaSStrassenMultiplicationFuncTests, MatrixMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {std::make_tuple(1, "2x2"),         std::make_tuple(2, "4x4"),
                                            std::make_tuple(3, "8x8"),         std::make_tuple(4, "16x16"),
                                            std::make_tuple(5, "32x32"),       std::make_tuple(6, "empty"),
                                            std::make_tuple(7, "invalid_size")};

const auto kTestTasksSEQ = ppc::util::AddFuncTask<MorozovaSStrassenMultiplicationSEQ, InType>(
    kTestParam, PPC_SETTINGS_morozova_s_strassen_multiplication);

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksSEQ);
const auto kPerfTestName =
    MorozovaSStrassenMultiplicationFuncTests::PrintFuncTestName<MorozovaSStrassenMultiplicationFuncTests>;

INSTANTIATE_TEST_SUITE_P(StrassenMultiplicationTests, MorozovaSStrassenMultiplicationFuncTests, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace morozova_s_strassen_multiplication
