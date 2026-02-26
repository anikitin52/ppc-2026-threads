#include <gtest/gtest.h>

// #include "nikitin_a_monte_carlo/all/include/ops_all.hpp"
#include "nikitin_a_monte_carlo/common/include/common.hpp"
// #include "nikitin_a_monte_carlo/omp/include/ops_omp.hpp"
#include "nikitin_a_monte_carlo/seq/include/ops_seq.hpp"
// #include "nikitin_a_monte_carlo/stl/include/ops_stl.hpp"
// #include "nikitin_a_monte_carlo/tbb/include/ops_tbb.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitin_a_monte_carlo {

class NikitinAPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 200;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NikitinAPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {


}  // namespace

}  // namespace nikitin_a_monte_carlo
