#include "nikitin_a_monte_carlo/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "nikitin_a_monte_carlo/common/include/common.hpp"
#include "util/include/util.hpp"

namespace nikitin_a_monte_carlo {

NikitinAMonteCarloSEQ::NikitinAMonteCarloSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool NikitinAMonteCarloSEQ::ValidationImpl() {
  return 0;
}

bool NikitinAMonteCarloSEQ::PreProcessingImpl() {
  return 0;
}

bool NikitinAMonteCarloSEQ::RunImpl() {
  return 0;
}

bool NikitinAMonteCarloSEQ::PostProcessingImpl() {
  return 0;
}

}  // namespace nikitin_a_monte_carlo
