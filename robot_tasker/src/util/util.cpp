#include "util.h"
#include <filesystem>

namespace util
{

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

float squaref(float x) noexcept
{
  return x*x;
}

std::string get_file_dir_path(const std::string& filename)
{
  std::filesystem::path file(filename);
  return file.parent_path().string();
}

} //namespace {util}