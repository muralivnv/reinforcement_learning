#ifndef _ANN_UTIL_H_
#define _ANN_UTIL_H_

#include <functional>

#include "../global_typedef.h"
#include "ANN_type_traits.h"
#include "ANN_typedef.h"
#include "ANN.h"

namespace ANN{

void gradient_clipping(const float threshold, eig::Ref<eig::ArrayXf> gradient)
{
  float gradient_norm = std::sqrtf(gradient.square().sum());
  if (gradient_norm > threshold)
  {
    gradient *= (threshold/gradient_norm);
  }
}

template<int InputSize, int ... NHiddenLayers>
void display_layer_weights_norm(const ANN::ArtificialNeuralNetwork<InputSize, NHiddenLayers ...>& network)
{
  int weights_start = 0;
  std::vector<float> w_norm(network.n_layers, 0.0F);

  for (int layer = 1u; layer <= (int)network.n_layers; layer++)
  {
    int n_nodes_last_layer = (int)network.n_nodes(layer-1u);
    int n_nodes_cur_layer  = (int)network.n_nodes(layer);
    int weights_end   = weights_start + (n_nodes_last_layer*n_nodes_cur_layer);

    auto this_layer_w = network.weight(seq(weights_start, weights_end-1));
    w_norm[layer-1] = std::sqrtf(this_layer_w.square().sum());

    weights_start  = weights_end;
  }

  for (size_t i = 0u; i < w_norm.size(); i++)
  {
    std::cout << i+1 << ": " << w_norm[i] << " | ";
  }
  std::cout << '\n';
}


// template<int InputSize, int ... NHiddenLayers>
// void normalize_if_req(const float norm_threshold, ANN::ArtificialNeuralNetwork<InputSize, NHiddenLayers ...>& network)
// {
//   int weights_start = 0;
//   for (int layer = 1u; layer <= (int)network.n_layers; layer++)
//   {
//     int n_nodes_last_layer = (int)network.n_nodes(layer-1u);
//     int n_nodes_cur_layer  = (int)network.n_nodes(layer);
//     int weights_end   = weights_start + (n_nodes_last_layer*n_nodes_cur_layer);

//     auto this_layer_w = network.weight(seq(weights_start, weights_end-1));
//     float this_layer_norm = std::sqrtf(this_layer_w.square().sum());
//     if (this_layer_norm > norm_threshold)
//     { 
//       this_layer_w /= this_layer_norm;
//     }

//     weights_start  = weights_end;
//   }
// }

} // namespace {ANN}

#endif
