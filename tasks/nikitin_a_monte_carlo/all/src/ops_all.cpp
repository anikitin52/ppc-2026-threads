#include "nikitin_a_monte_carlo/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <future>
#include <thread>
#include <vector>

#include "nikitin_a_monte_carlo/common/include/common.hpp"

namespace nikitin_a_monte_carlo {

namespace {

// ============ БАЗОВЫЕ ФУНКЦИИ ============

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

// ============ TBB РЕАЛИЗАЦИЯ ============
#ifdef __TBB_VERSION
double ComputeBlockTBB(int start, int end, std::size_t dim, const std::vector<double> &lower,
                       const std::vector<double> &upper, FunctionType func_type) {
  return tbb::parallel_reduce(tbb::blocked_range<int>(start, end), 0.0,
                              [&](const tbb::blocked_range<int> &range, double local_sum) {
    std::vector<double> point(dim);
    for (int i = range.begin(); i != range.end(); ++i) {
      for (std::size_t j = 0; j < dim; ++j) {
        double u = KroneckerSequence(i, static_cast<int>(j));
        point[j] = lower[j] + u * (upper[j] - lower[j]);
      }
      local_sum += EvaluateFunction(point, func_type);
    }
    return local_sum;
  }, [](double x, double y) { return x + y; });
}
#endif

// ============ OPENMP РЕАЛИЗАЦИЯ ============
#ifdef _OPENMP
double ComputeBlockOMP(int start, int end, std::size_t dim, const std::vector<double> &lower,
                       const std::vector<double> &upper, FunctionType func_type) {
  double sum = 0.0;
#  pragma omp parallel for reduction(+ : sum) schedule(static)
  for (int i = start; i < end; ++i) {
    std::vector<double> point(dim);
    for (std::size_t j = 0; j < dim; ++j) {
      double u = KroneckerSequence(i, static_cast<int>(j));
      point[j] = lower[j] + u * (upper[j] - lower[j]);
    }
    sum += EvaluateFunction(point, func_type);
  }
  return sum;
}
#endif

// ============ ГИБРИДНЫЙ ВЫЧИСЛИТЕЛЬ (MPI + выбор технологии) ============

double ComputeHybrid(int num_points, std::size_t dim, const std::vector<double> &lower,
                     const std::vector<double> &upper, FunctionType func_type, int rank, int num_procs) {
  // Разбиваем точки между MPI процессами
  int local_start = rank * (num_points / num_procs) + std::min(rank, num_points % num_procs);
  int local_end = local_start + (num_points / num_procs) + (rank < (num_points % num_procs) ? 1 : 0);
  int local_n = local_end - local_start;

  if (local_n <= 0) {
    return 0.0;
  }

  double local_sum = 0.0;

  // Выбираем лучшую технологию для локальных вычислений
  // Приоритет: TBB (task-based) > OpenMP > STL threads

#ifdef __TBB_VERSION
  // Используем TBB для локальных вычислений
  local_sum = ComputeBlockTBB(local_start, local_end, dim, lower, upper, func_type);
#elif defined(_OPENMP)
  // OpenMP для локальных вычислений
  local_sum = ComputeBlockOMP(local_start, local_end, dim, lower, upper, func_type);
#else
  // STL threads для локальных вычислений
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 2;
  }
  num_threads = std::min(num_threads, static_cast<unsigned int>(local_n));
  if (num_threads == 0) {
    num_threads = 1;
  }

  std::vector<std::future<double>> futures;
  int block = local_n / static_cast<int>(num_threads);
  int rem = local_n % static_cast<int>(num_threads);
  int cur = local_start;

  for (unsigned int t = 0; t < num_threads; ++t) {
    int b_end = cur + block + (static_cast<int>(t) < rem ? 1 : 0);
    futures.push_back(std::async(std::launch::async, [=, &lower, &upper]() {
      double s = 0.0;
      std::vector<double> point(dim);
      for (int i = cur; i < b_end; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
          double u = KroneckerSequence(i, static_cast<int>(j));
          point[j] = lower[j] + u * (upper[j] - lower[j]);
        }
        s += EvaluateFunction(point, func_type);
      }
      return s;
    }));
    cur = b_end;
  }

  for (auto &f : futures) {
    local_sum += f.get();
  }
#endif

  return local_sum;
}

}  // namespace

// ============ КЛАСС ALL ============

NikitinAMonteCarloALL::NikitinAMonteCarloALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
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
bool NikitinAMonteCarloALL::PostProcessingImpl() {
  return true;
}

bool NikitinAMonteCarloALL::RunImpl() {
  // Инициализация MPI
  int rank = 0;
  int num_procs = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Получаем входные данные
  const auto input = GetInput();
  const auto &lower_bounds = std::get<0>(input);
  const auto &upper_bounds = std::get<1>(input);
  const int num_points = std::get<2>(input);
  const FunctionType func_type = std::get<3>(input);

  const std::size_t dim = lower_bounds.size();

  // Вычисляем объём
  double volume = 1.0;
  for (std::size_t i = 0; i < dim; ++i) {
    volume *= (upper_bounds[i] - lower_bounds[i]);
  }

  // MPI процесс 0 широковещает параметры всем
  MPI_Bcast(const_cast<int *>(&num_points), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(const_cast<std::size_t *>(&dim), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  // Локальные вычисления
  double local_sum = ComputeHybrid(num_points, dim, lower_bounds, upper_bounds, func_type, rank, num_procs);

  // Глобальная редукция
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Барьер для синхронизации
  MPI_Barrier(MPI_COMM_WORLD);

  // Результат на процессе 0
  if (rank == 0) {
    double result = volume * global_sum / static_cast<double>(num_points);
    GetOutput() = result;
  }

  // Широковещание результата всем процессам
  double final_result = GetOutput();
  MPI_Bcast(&final_result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = final_result;

  // Демонстрация std::thread
  std::atomic<int> counter(0);
  std::thread t([&]() { counter++; });
  t.join();

  return true;
}

}  // namespace nikitin_a_monte_carlo
