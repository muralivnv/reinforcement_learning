#ifndef _ANN_TYPEDEF_H_
#define _ANN_TYPEDEF_H_

#include <unordered_map>
#include <any>
#include <memory>

#include "../global_typedef.h"
#include "ANN_activation.h"
#include "ANN_initialization.h"

namespace ANN
{
using weights_t      = eig::Array<float, eig::Dynamic, 1>;
using bias_t         = eig::Array<float, eig::Dynamic, 1>;
using activation_t   = std::unique_ptr<ANN::ActivationBase>;
using initializer_t  = std::unique_ptr<ANN::ParamInitializerBase>;

template<int N>
using activation_list_t = std::array<activation_t, N>;

template<int N>
using initializer_list_t = std::array<initializer_t, N>;

template<int N>
using layer_config_t = std::array<size_t, N>;

using OptimizerParams = std::unordered_map<std::string, std::any>;

} // namespace {ANN}
#endif
