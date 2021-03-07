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
using weights_t = RL::Arrayf<ann_weights_len<InputSize, LayerNodeConfig...>::value>;

template<int ...LayerNodeConfig>
using bias_t = RL::Arrayf<pack_add<LayerNodeConfig...>::value>;

template<int ...LayerNodeConfig>
using activations_t  = std::array<ANN::ActivationBase, pack_len<LayerNodeConfig...>::value>;

template<int InputSize, int ...LayerNodeConfig>
using layerConfig_t = RL::Array<size_t, pack_len<InputSize, LayerNodeConfig...>::value>;

using lossFcn_t     = std::function<RL::MatrixX<float>(const RL::MatrixX<float>&, const RL::MatrixX<float>&)>;
using lossFcnGrad_t = std::function<RL::MatrixX<float>(const RL::MatrixX<float>&, const RL::MatrixX<float>&)>;

template<int ...LayerNodeConfig>
using output_t = RL::Arrayf<ann_output_len<LayerNodeConfig...>::value>;

template<int SampleSize, int ...LayerNodeConfig>
using output_batch_t = RL::Matrix<float, SampleSize, ann_output_len<LayerNodeConfig...>::value>;

using OptimizerParams = std::unordered_map<std::string, std::any>;

} // namespace {ANN}
#endif
