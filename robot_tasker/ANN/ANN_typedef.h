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

template<int OutputLen>
constexpr int storage_order()
{
  if constexpr (OutputLen == 1)
  { return eig::ColMajor; }
  else 
  { return eig::RowMajor; }
}

template<int InputSize, int ... NHiddenLayers>
using weights_t = eig::Array<float, ann_weights_len<InputSize, NHiddenLayers...>::value, 1>;

template<int ...NHiddenLayers>
using bias_t = eig::Array<float, pack_add<NHiddenLayers...>::value, 1>;

template<int ...NHiddenLayers>
using activations_t  = std::array<ANN::Activation, pack_len<NHiddenLayers...>::value>;

template<int InputSize, int ...NHiddenLayers>
using layerConfig_t = eig::Array<int, 1, pack_len<InputSize, NHiddenLayers...>::value>;

template<typename EigenDerived1, typename EigenDerived2>
using lossFcn_t     = std::function< eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>(const eig::ArrayBase<EigenDerived1>&, const eig::ArrayBase<EigenDerived2>&)>;

template<typename EigenDerived1, typename EigenDerived2>
using lossFcnGrad_t = std::function< eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>(const eig::ArrayBase<EigenDerived1>&, const eig::ArrayBase<EigenDerived2>&)>;

template<int ...NHiddenLayers>
using output_t = eig::Array<float, ann_output_len<NHiddenLayers...>::value, 1>;

template<int SampleSize, int ...NHiddenLayers>
using output_batch_t = eig::Array<float, SampleSize, ann_output_len<NHiddenLayers...>::value, storage_order<ann_output_len<NHiddenLayers...>::value>()>;//eig::RowMajor>;

using OptimizerParams = std::unordered_map<std::string, std::any>;

} // namespace {ANN}
#endif
