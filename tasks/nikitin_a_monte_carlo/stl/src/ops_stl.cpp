#include "nikitin_a_monte_carlo/stl/include/ops_stl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

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
  return std::fmod(index * alpha, 1.0);
}

// Функция для вычисления суммы на отрезке [start, end)
double ComputePartialSum(int start, int end, std::size_t dim, const std::vector<double> &lower_bounds,
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

}  // namespace

NikitinAMonteCarloSTL::NikitinAMonteCarloSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool NikitinAMonteCarloSTL::ValidationImpl() {
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

bool NikitinAMonteCarloSTL::PreProcessingImpl() {
  return true;
}

bool NikitinAMonteCarloSTL::RunImpl() {
  // Получаем входные данные и распаковываем их в обычные переменные
  const auto input = GetInput();
  const auto &lower_bounds = std::get<0>(input);
  const auto &upper_bounds = std::get<1>(input);
  const int num_points = std::get<2>(input);
  const FunctionType func_type = std::get<3>(input);

  std::size_t dim = lower_bounds.size();

  // Вычисление объема области интегрирования
  double volume = 1.0;
  for (std::size_t i = 0; i < dim; ++i) {
    volume *= (upper_bounds[i] - lower_bounds[i]);
  }

  // Определяем количество потоков (по умолчанию - аппаратные ядра)
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 2;  // fallback
  }

  // Не создаем больше потоков, чем точек
  if (num_threads > static_cast<unsigned int>(num_points)) {
    num_threads = static_cast<unsigned int>(num_points);
  }

  // Разбиваем работу на части
  std::vector<std::future<double>> futures;
  int points_per_thread = num_points / num_threads;
  int remainder = num_points % num_threads;

  int current_start = 0;

  for (unsigned int t = 0; t < num_threads; ++t) {
    int start = current_start;
    int end = start + points_per_thread + (t < static_cast<unsigned int>(remainder) ? 1 : 0);
    current_start = end;

    // Запускаем асинхронное вычисление
    futures.push_back(std::async(std::launch::async, ComputePartialSum, start, end, dim, std::cref(lower_bounds),
                                 std::cref(upper_bounds), func_type));
  }

  // Собираем результаты
  double sum = 0.0;
  for (auto &future : futures) {
    sum += future.get();
  }

  double result = volume * sum / static_cast<double>(num_points);
  GetOutput() = result;

  return true;
}

bool NikitinAMonteCarloSTL::PostProcessingImpl() {
  return true;
}

}  // namespace nikitin_a_monte_carlo
