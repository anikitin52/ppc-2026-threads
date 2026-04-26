#include "nikitin_a_monte_carlo/all/include/ops_all.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <future>
#include <thread>
#include <vector>

// Проверка доступности технологий
#if defined(_OPENMP)
#  include <omp.h>
#  define HAS_OMP 1
#else
#  define HAS_OMP 0
#endif

#ifdef __TBB_VERSION
#  include <oneapi/tbb.h>
#  define HAS_TBB 1
#else
#  define HAS_TBB 0
#endif

#include "nikitin_a_monte_carlo/common/include/common.hpp"

namespace nikitin_a_monte_carlo {

namespace {
// Вспомогательная функция для вычисления значения тестовой функции
double EvaluateFunction(const std::vector<double> &point, FunctionType type) {
  if (point.empty()) {
    return 0.0;
  }

  switch (type) {
    case FunctionType::kConstant:
      return 1.0;
    case FunctionType::kLinear:
      return point.at(0);
    case FunctionType::kProduct:
      if (point.size() < 2) {
        return 0.0;
      }
      return point.at(0) * point.at(1);
    case FunctionType::kQuadratic:
      if (point.size() < 2) {
        return 0.0;
      }
      return (point.at(0) * point.at(0)) + (point.at(1) * point.at(1));
    case FunctionType::kExponential:
      return std::exp(point.at(0));
    default:
      return 0.0;
  }
}

// Генерация квазислучайной последовательности Кронекера
double KroneckerSequence(int index, int dimension) {
  const std::array<double, 10> primes = {2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0};
  double alpha = std::sqrt(primes.at(static_cast<std::size_t>(dimension % 10)));
  alpha = alpha - std::floor(alpha);
  return std::fmod(static_cast<double>(index) * alpha, 1.0);
}

// ============ SEQ РЕАЛИЗАЦИЯ ============
double ComputeMonteCarloSEQ(int num_points, std::size_t dim, const std::vector<double> &lower_bounds,
                            const std::vector<double> &upper_bounds, FunctionType func_type) {
  double sum = 0.0;
  std::vector<double> point(dim);

  for (int i = 0; i < num_points; ++i) {
    for (std::size_t j = 0; j < dim; ++j) {
      double u = KroneckerSequence(i, static_cast<int>(j));
      point[j] = lower_bounds[j] + (u * (upper_bounds[j] - lower_bounds[j]));
    }
    sum += EvaluateFunction(point, func_type);
  }

  return sum;
}

// ============ OMP РЕАЛИЗАЦИЯ ============
#if HAS_OMP
double ComputeMonteCarloOMP(int num_points, std::size_t dim, const std::vector<double> &lower_bounds,
                            const std::vector<double> &upper_bounds, FunctionType func_type) {
  double sum = 0.0;

#  pragma omp parallel for reduction(+ : sum) schedule(static)
  for (int i = 0; i < num_points; ++i) {
    std::vector<double> point(dim);
    for (std::size_t j = 0; j < dim; ++j) {
      double u = KroneckerSequence(i, static_cast<int>(j));
      point[j] = lower_bounds[j] + (u * (upper_bounds[j] - lower_bounds[j]));
    }
    sum += EvaluateFunction(point, func_type);
  }

  return sum;
}
#endif

// ============ TBB РЕАЛИЗАЦИЯ ============
#if HAS_TBB
double ComputeMonteCarloTBB(int num_points, std::size_t dim, const std::vector<double> &lower_bounds,
                            const std::vector<double> &upper_bounds, FunctionType func_type) {
  return tbb::parallel_reduce(tbb::blocked_range<int>(0, num_points), 0.0,
                              [&](const tbb::blocked_range<int> &range, double local_sum) -> double {
    std::vector<double> point(dim);
    for (int i = range.begin(); i != range.end(); ++i) {
      for (std::size_t j = 0; j < dim; ++j) {
        double u = KroneckerSequence(i, static_cast<int>(j));
        point[j] = lower_bounds[j] + (u * (upper_bounds[j] - lower_bounds[j]));
      }
      local_sum += EvaluateFunction(point, func_type);
    }
    return local_sum;
  }, [](double x, double y) -> double { return x + y; });
}
#endif

// ============ STL РЕАЛИЗАЦИЯ ============
double ComputePartialSumSTL(int start, int end, std::size_t dim, const std::vector<double> &lower_bounds,
                            const std::vector<double> &upper_bounds, FunctionType func_type) {
  double local_sum = 0.0;
  std::vector<double> point(dim);

  for (int i = start; i < end; ++i) {
    for (std::size_t j = 0; j < dim; ++j) {
      double u = KroneckerSequence(i, static_cast<int>(j));
      point[j] = lower_bounds[j] + (u * (upper_bounds[j] - lower_bounds[j]));
    }
    local_sum += EvaluateFunction(point, func_type);
  }

  return local_sum;
}

double ComputeMonteCarloSTL(int num_points, std::size_t dim, const std::vector<double> &lower_bounds,
                            const std::vector<double> &upper_bounds, FunctionType func_type) {
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 2U;
  }

  const auto unsigned_num_points = static_cast<unsigned int>(num_points);
  num_threads = std::min(num_threads, unsigned_num_points);

  if (num_threads == 0) {
    num_threads = 1U;
  }

  std::vector<std::future<double>> futures;
  const int points_per_thread = num_points / static_cast<int>(num_threads);
  const int remainder = num_points % static_cast<int>(num_threads);

  int current_start = 0;

  for (unsigned int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    const int start = current_start;
    const int thread_idx_signed = static_cast<int>(thread_idx);
    const int end = start + points_per_thread + ((thread_idx_signed < remainder) ? 1 : 0);
    current_start = end;

    futures.push_back(
        std::async(std::launch::async, ComputePartialSumSTL, start, end, dim, lower_bounds, upper_bounds, func_type));
  }

  double sum = 0.0;
  for (auto &future : futures) {
    sum += future.get();
  }

  return sum;
}

}  // namespace

// ============ КЛАСС ALL ============
NikitinAMonteCarloALL::NikitinAMonteCarloALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

ppc::task::TypeOfTask NikitinAMonteCarloALL::SelectBestTechnology() const {
// Приоритет выбора: TBB → OpenMP → STL → SEQ
#if HAS_TBB
  return ppc::task::TypeOfTask::kTBB;
#elif HAS_OMP
  return ppc::task::TypeOfTask::kOMP;
#else
  // STL доступен всегда (C++11 и выше)
  return ppc::task::TypeOfTask::kSTL;
#endif
}

bool NikitinAMonteCarloALL::ValidationImpl() {
  const auto &[lower_bounds, upper_bounds, num_points, func_type] = GetInput();

  if (lower_bounds.empty() || upper_bounds.empty()) {
    return false;
  }

  if (lower_bounds.size() != upper_bounds.size()) {
    return false;
  }

  for (std::size_t i = 0; i < lower_bounds.size(); ++i) {
    if (lower_bounds[i] >= upper_bounds[i]) {
      return false;
    }
  }

  return num_points > 0;
}

bool NikitinAMonteCarloALL::PreProcessingImpl() {
  return true;
}

bool NikitinAMonteCarloALL::RunImpl() {
  const auto input = GetInput();
  const auto &lower_bounds = std::get<0>(input);
  const auto &upper_bounds = std::get<1>(input);
  const int num_points = std::get<2>(input);
  const FunctionType func_type = std::get<3>(input);

  const std::size_t dim = lower_bounds.size();

  // Вычисление объема области интегрирования
  double volume = 1.0;
  for (std::size_t i = 0; i < dim; ++i) {
    volume *= (upper_bounds[i] - lower_bounds[i]);
  }

  // Автоматический выбор технологии
  const auto best_tech = SelectBestTechnology();
  double sum = 0.0;

  switch (best_tech) {
    case ppc::task::TypeOfTask::kTBB:
#if HAS_TBB
      sum = ComputeMonteCarloTBB(num_points, dim, lower_bounds, upper_bounds, func_type);
#endif
      break;

    case ppc::task::TypeOfTask::kOMP:
#if HAS_OMP
      sum = ComputeMonteCarloOMP(num_points, dim, lower_bounds, upper_bounds, func_type);
#endif
      break;

    case ppc::task::TypeOfTask::kSTL:
      sum = ComputeMonteCarloSTL(num_points, dim, lower_bounds, upper_bounds, func_type);
      break;

    default:
      sum = ComputeMonteCarloSEQ(num_points, dim, lower_bounds, upper_bounds, func_type);
      break;
  }

  const double result = volume * sum / static_cast<double>(num_points);
  GetOutput() = result;

  return true;
}

bool NikitinAMonteCarloALL::PostProcessingImpl() {
  return true;
}

}  // namespace nikitin_a_monte_carlo
