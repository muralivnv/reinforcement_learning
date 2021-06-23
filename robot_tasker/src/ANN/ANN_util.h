#ifndef _ANN_UTIL_H_
#define _ANN_UTIL_H_

#include <functional>

#include "../global_typedef.h"
#include "ANN_type_traits.h"
#include "ANN_typedef.h"

namespace ANN{

// template<int N>
void gradient_clipping(const float threshold, eig::Ref<eig::ArrayXf> gradient)
{
  float gradient_norm = std::sqrtf(gradient.square().sum());
  if (gradient_norm > threshold)
  {
    gradient *= (threshold/gradient_norm);
  }
}

} // namespace {ANN}

#endif
