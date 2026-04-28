#include "nikitin_a_monte_carlo/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <future>
#include <thread>
#include <vector>

#include "nikitin_a_monte_carlo/common/include/common.hpp"

namespace nikitin_a_monte_carlo {

namespace {
double EvaluateFunction(const std::vector<double> &point, FunctionType type) {
  if (point.empty()) {
    return 0.0;
  }
  switch (type) {
    case FunctionType::kConstant:
      return 1.0;
    case FunctionType::kLinear:
      return point[0];
    case FunctionType::kProduct:
      return (point.size() >= 2) ? point[0] * point[1] : 0.0;
    case FunctionType::kQuadratic:
      return (point.size() >= 2) ? point[0] * point[0] + point[1] * point[1] : 0.0;
    case FunctionType::kExponential:
      return std::exp(point[0]);
    default:
      return 0.0;
  }
}

double KroneckerSequence(int index, int dimension) {
  const std::array<double, 10> primes = {2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0};
  double alpha = std::sqrt(primes[dimension % 10]);
  alpha = alpha - std::floor(alpha);
  return std::fmod(static_cast<double>(index) * alpha, 1.0);
}

// Вычисление одного блока точек (используется на всех уровнях)
double ComputeBlock(int start, int end, std::size_t dim, const std::vector<double> &lower,
                    const std::vector<double> &upper, FunctionType func) {
  double sum = 0.0;
  std::vector<double> point(dim);
  for (int i = start; i < end; ++i) {
    for (std::size_t j = 0; j < dim; ++j) {
      double u = KroneckerSequence(i, static_cast<int>(j));
      point[j] = lower[j] + u * (upper[j] - lower[j]);
    }
    sum += EvaluateFunction(point, func);
  }
  return sum;
}

double ComputeBlockTBB(int start, int end, std::size_t dim, const std::vector<double> &lower,
                       const std::vector<double> &upper, FunctionType func) {
  return tbb::parallel_reduce(tbb::blocked_range<int>(start, end), 0.0,
                              [&](const tbb::blocked_range<int> &r, double local) {
    std::vector<double> point(dim);
    for (int i = r.begin(); i != r.end(); ++i) {
      for (std::size_t j = 0; j < dim; ++j) {
        double u = KroneckerSequence(i, static_cast<int>(j));
        point[j] = lower[j] + u * (upper[j] - lower[j]);
      }
      local += EvaluateFunction(point, func);
    }
    return local;
  }, [](double x, double y) { return x + y; });
}

double ComputeBlockOMP(int start, int end, std::size_t dim, const std::vector<double> &lower,
                       const std::vector<double> &upper, FunctionType func) {
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (int i = start; i < end; ++i) {
    std::vector<double> point(dim);
    for (std::size_t j = 0; j < dim; ++j) {
      double u = KroneckerSequence(i, static_cast<int>(j));
      point[j] = lower[j] + u * (upper[j] - lower[j]);
    }
    sum += EvaluateFunction(point, func);
  }
  return sum;
}

double ComputeBlockSTL(int start, int end, std::size_t dim, const std::vector<double> &lower,
                       const std::vector<double> &upper, FunctionType func) {
  if (start >= end) {
    return 0.0;
  }

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 2U;
  }

  int total = end - start;
  num_threads = std::min(num_threads, static_cast<unsigned int>(total));
  if (num_threads == 0) {
    num_threads = 1U;
  }

  std::vector<std::future<double>> futures;
  int pts_per_thread = total / static_cast<int>(num_threads);
  int remainder = total % static_cast<int>(num_threads);
  int current = start;

  for (unsigned int tid = 0; tid < num_threads; ++tid) {
    int block_end = current + pts_per_thread + (static_cast<int>(tid) < remainder ? 1 : 0);
    futures.push_back(std::async(std::launch::async, ComputeBlock, current, block_end, dim, lower, upper, func));
    current = block_end;
  }

  double sum = 0.0;
  for (auto &f : futures) {
    sum += f.get();
  }
  return sum;
}

// Оптимальная стратегия для одного процесса MPI
double ComputeOnProcess(int num_points, std::size_t dim, const std::vector<double> &lower,
                        const std::vector<double> &upper, FunctionType func) {
  // Автовыбор: TBB (task-based) > OMP (loop-based) > STL (threads)

  int min_block_size = 10000;  // Порог для переключения
  bool use_tbb = (num_points > min_block_size);

  if (num_points < 1000) {
    return ComputeBlock(0, num_points, dim, lower, upper, func);
  }

#ifdef __TBB_VERSION
  if (use_tbb) {
    return ComputeBlockTBB(0, num_points, dim, lower, upper, func);
  }
#endif

#ifdef _OPENMP
  if (use_tbb && num_points > 50000) {
    return ComputeBlockOMP(0, num_points, dim, lower, upper, func);
  }
#endif

  return ComputeBlockSTL(0, num_points, dim, lower, upper, func);
}

}  // namespace

NikitinAMonteCarloALL::NikitinAMonteCarloALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool NikitinAMonteCarloALL::ValidationImpl() {
  const auto &[low, up, n, func] = GetInput();
  if (low.empty() || up.empty() || low.size() != up.size()) {
    return false;
  }
  for (std::size_t i = 0; i < low.size(); ++i) {
    if (low[i] >= up[i]) {
      return false;
    }
  }
  return n > 0;
}

bool NikitinAMonteCarloALL::PreProcessingImpl() {
  return true;
}
bool NikitinAMonteCarloALL::PostProcessingImpl() {
  return true;
}

bool NikitinAMonteCarloALL::RunImpl() {
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto input = GetInput();
  const auto &lower = std::get<0>(input);
  const auto &upper = std::get<1>(input);
  int total_points = std::get<2>(input);
  const FunctionType func_type = std::get<3>(input);

  const std::size_t dim = lower.size();

  // Вычисление объема
  double volume = 1.0;
  for (std::size_t i = 0; i < dim; ++i) {
    volume *= (upper[i] - lower[i]);
  }

  // === MPI: распределение точек между процессами ===
  int points_per_proc = total_points / world_size;
  int remainder_points = total_points % world_size;

  int start = rank * points_per_proc + std::min(rank, remainder_points);
  int end = start + points_per_proc + (rank < remainder_points ? 1 : 0);
  int local_points = end - start;

  // === Локальные вычисления (с использованием лучшей технологии) ===
  double local_sum = ComputeOnProcess(local_points, dim, lower, upper, func_type);

  // === MPI: сбор результатов ===
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double result = 0.0;
  if (rank == 0) {
    result = volume * global_sum / static_cast<double>(total_points);
    GetOutput() = result;
  }

  // Синхронизация
  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput() = result;
  }

  return true;
}

}  // namespace nikitin_a_monte_carlo
