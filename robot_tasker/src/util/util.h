#ifndef _UTIL_H_
#define _UTIL_H_

#include "../global_typedef.h"

namespace util
{

float linear_interpolate(const float x, 
                         const float x1, const float y1, 
                         const float x2, const float y2);

float wrapto_minuspi_pi(float angle) noexcept;

float squaref(float x) noexcept;

template<typename ArgLeft, typename ... ArgsRight>
inline auto min (ArgLeft x, ArgsRight ... y) noexcept
{
  typename std::common_type_t<ArgLeft, ArgsRight...> result = x;
  ((result = ((result < y)? result : y) ), ...);
  
  return result;
}

template<typename ArgLeft, typename ... ArgsRight>
inline auto max (ArgLeft x, ArgsRight ... y) noexcept
{
  typename std::common_type_t<ArgLeft, ArgsRight...> result = x;
  ((result = ((result > y)? result : y) ), ...);
  
  return result;
}

template<int N>
std::array<int, N> get_n_shuffled_indices(const int container_size)
{
  std::vector<int> indices(container_size);
  std::array<int, N> shuffled_n_indices;
  std::random_device rd;
  std::mt19937 g(rd());

  std::generate(indices.begin(), indices.end(), [n = 0] () mutable { return n++; });
  std::shuffle(indices.begin(), indices.end(), g);
  std::copy_n(indices.cbegin(), N, shuffled_n_indices.begin());

  return shuffled_n_indices;
}


std::string get_file_dir_path(const std::string& filename);

std::unordered_map<std::string, float>
read_global_config(const std::string& config_name);

} // namespace {util}

#endif
