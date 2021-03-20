#ifndef _UTIL_H_
#define _UTIL_H_

#include "global_typedef.h"

namespace RL
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

float wrapto_minuspi_pi(float angle)
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

} // namespace {RL}
#endif
