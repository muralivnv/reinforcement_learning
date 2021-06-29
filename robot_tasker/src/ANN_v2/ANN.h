#ifndef _ANN_H_
#define _ANN_H_

#include "ANN_typedef.h"

namespace ANN
{

template<int NLayers>
struct ArtificialNeuralNetwork
{
  ANN::weights_t                     weights;
  ANN::bias_t                        bias;
  ANN::activation_list_t<NLayers-1>  activations;
  ANN::initializer_list_t<NLayers-1> initializers;
  ANN::layer_config_t<NLayers>       n_nodes;
};

} //namespace {ANN}
#endif
