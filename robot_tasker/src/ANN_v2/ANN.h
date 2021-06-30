#ifndef _ANN_H_
#define _ANN_H_

#include "ANN_typedef.h"

namespace ann
{

template<int NLayers>
struct ArtificialNeuralNetwork
{
  weights_t                     weights;
  bias_t                        bias;
  activation_list_t<NLayers-1>  activations;
  initializer_list_t<NLayers-1> initializers;
  layer_config_t<NLayers>       n_nodes;
};

} //namespace {ann}
#endif
