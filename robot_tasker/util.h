#ifndef _UTIL_H_
#define _UTIL_H_

namespace RL
{
float linear_interpolate(const float x, 
                         const float x1, const float y1, 
                         const float x2, const float y2, 
                         const bool is_bounded=true)
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

} // namespace {RL}
#endif
