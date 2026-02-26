#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// #include "nikitin_a_monte_carlo/all/include/ops_all.hpp"
#include "nikitin_a_monte_carlo/common/include/common.hpp"
// #include "nikitin_a_monte_carlo/omp/include/ops_omp.hpp"
#include "nikitin_a_monte_carlo/seq/include/ops_seq.hpp"
// #include "nikitin_a_monte_carlo/stl/include/ops_stl.hpp"
// #include "nikitin_a_monte_carlo/tbb/include/ops_tbb.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nikitin_a_monte_carlo {

class NikitinAFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    return;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() final {
    return 0;
  }

 private:
  InType input_data_ = 0;
};

namespace {


}  // namespace

}  // namespace nikitin_a_monte_carlo
