#ifndef _UTIL_H_
#define _UTIL_H_

#include "global_typedef.h"

namespace RL
{

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

float linear_interpolate(const float x, 
                         const float x1, const float y1, 
                         const float x2, const float y2)
{
  float interpolated_val;

  if (x < x1)
  {  interpolated_val = y1;  }
  else if (x > x2)
  {  interpolated_val = y2;  }
  else
  {
    interpolated_val = y1 + (x - x1)*(y2 - y1)/(x2 - x1);
  }
  return interpolated_val;
}

float wrapto_minuspi_pi(float angle) noexcept
{
  float normalized_angle = fmodf(angle, TWO_PI);
  if (PI < normalized_angle)
  {
    normalized_angle -= TWO_PI;
  }
  else if (normalized_angle < -PI)
  {
    normalized_angle += TWO_PI;
  }
  return normalized_angle;
}

inline float squaref(float x) noexcept
{
  return x*x;
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

} // namespace {RL}
#endif
