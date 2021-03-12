#ifndef _ANN_TYPEDEF_H_
#define _ANN_TYPEDEF_H_

#include <functional>
#include <unordered_map>
#include <any>

#include "../global_typedef.h"
#include "ANN_type_traits.h"
#include "ANN_activation.h"

namespace ANN
{

template<int InputSize, int ... LayerNodeConfig>
using weights_t = eig::Array<float, ann_weights_len<InputSize, LayerNodeConfig...>::value, 1>;

template<int ...LayerNodeConfig>
using bias_t = eig::Array<float, pack_add<LayerNodeConfig...>::value, 1>;

template<int ...LayerNodeConfig>
using activations_t  = std::array<ANN::ActivationBase, pack_len<LayerNodeConfig...>::value>;

template<int InputSize, int ...LayerNodeConfig>
using layerConfig_t = eig::Array<int, 1, pack_len<InputSize, LayerNodeConfig...>::value>;

template<typename EigenDerived1, typename EigenDerived2>
using lossFcn_t     = std::function< eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>(const eig::ArrayBase<EigenDerived1>&, const eig::ArrayBase<EigenDerived2>&)>;

template<typename EigenDerived1, typename EigenDerived2>
using lossFcnGrad_t = std::function< eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>(const eig::ArrayBase<EigenDerived1>&, const eig::ArrayBase<EigenDerived2>&)>;

template<int ...LayerNodeConfig>
using output_t = eig::Array<float, ann_output_len<LayerNodeConfig...>::value, 1>;

template<int SampleSize, int ...LayerNodeConfig>
using output_batch_t = eig::Array<float, SampleSize, ann_output_len<LayerNodeConfig...>::value>;

using OptimizerParams = std::unordered_map<std::string, std::any>;

} // namespace {ANN}
#endif
