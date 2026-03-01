#include "titaev_m_sortirovka_betchera/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstring>
#include <vector>
#include <cstdint>
#include "titaev_m_sortirovka_betchera/common/include/common.hpp"
namespace titaev_m_sortirovka_betchera {

TitaevSortirovkaBetcheraSEQ::TitaevSortirovkaBetcheraSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool TitaevSortirovkaBetcheraSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool TitaevSortirovkaBetcheraSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool TitaevSortirovkaBetcheraSEQ::RunImpl() {
  auto &result = GetOutput();
  const size_t n = result.size();
  if (n <= 1) {
    return true;
  }

  std::vector<uint64_t> keys(n);
  for (size_t i = 0; i < n; i++) {
    uint64_t x = 0;
    std::memcpy(&x, &result[i], sizeof(double));

    int64_t sx = static_cast<int64_t>(x);
    uint64_t mask = (static_cast<uint64_t>(sx >> 63) & 0x7FFFFFFFFFFFFFFFULL);
    x ^= mask;

    keys[i] = x;
  }

  std::vector<uint64_t> tmp(n);
  const int BITS = 8;
  const int BUCKETS = 1 << BITS;
  const int PASSES = 64 / BITS;

  for (int pass = 0; pass < PASSES; pass++) {
    std::vector<size_t> count(BUCKETS, 0);

    for (size_t i = 0; i < n; i++) {
      size_t bucket = (keys[i] >> (pass * BITS)) & (BUCKETS - 1);
      count[bucket]++;
    }

    for (int i = 1; i < BUCKETS; i++) {
      count[i] += count[i - 1];
    }

    for (size_t i = n; i-- > 0;) {
      size_t bucket = (keys[i] >> (pass * BITS)) & (BUCKETS - 1);
      tmp[--count[bucket]] = keys[i];
    }

    keys.swap(tmp);
  }

  for (size_t i = 0; i < n; i++) {
    uint64_t x = keys[i];

    int64_t sx = static_cast<int64_t>(x);
    uint64_t mask = (static_cast<uint64_t>((sx >> 63) - 1) & 0x7FFFFFFFFFFFFFFFULL);
    x ^= mask;

    std::memcpy(&result[i], &x, sizeof(double));
  }

  for (size_t p = 1; p < n; p <<= 1) {
    for (size_t k = p; k > 0; k >>= 1) {
      for (size_t i = 0; i < n; i++) {
        size_t j = i ^ k;
        if (j > i && j < n) {
          if ((i & p) == 0) {
            if (result[i] > result[j]) {
              std::swap(result[i], result[j]);
            }
          } else {
            if (result[i] < result[j]) {
              std::swap(result[i], result[j]);
            }
          }
        }
      }
    }
  }

  return true;
}

bool TitaevSortirovkaBetcheraSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
