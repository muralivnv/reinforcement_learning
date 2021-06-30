#ifndef _ANN_UTIL_H_
#define _ANN_UTIL_H_

#include <memory>

#include "ANN_typedef.h"
#include "ANN.h"

#include "ANN_activation.h"
#include "ANN_initialization.h"

namespace ANN
{

template<int N, typename ... T>
void set_layer_config(ArtificialNeuralNetwork<N>& ann, 
                      T... nodes)
{
  int index = 0;
  ((ann.n_nodes[index++] = nodes), ...);
  malloc_params(ann);
}

template<int N>
void malloc_params(ArtificialNeuralNetwork<N>& ann)
{
  const size_t n_bias = std::accumulate(ann.n_nodes.begin()+1u, ann.n_nodes.end(), (size_t)0u);
  
  size_t n_weights = 0u;
  for (size_t i = 1u; i < (size_t)N; i++)
  { n_weights += (ann.n_nodes[i-1]*ann.n_nodes[i]); }

  ann.weights.resize((int)n_weights, 1);
  ann.bias.resize((int)n_bias, 1);
}

template<int N, typename ... T>
void set_activations(ArtificialNeuralNetwork<N>& ann, 
                     std::unique_ptr<T>&& ... activations)
{
  static_assert( (N-1) == sizeof...(activations), "number_of_activations_not_equal_to_number_of_hidden_layers");

  int index = 0u;
  ((ann.activations[index++] = std::move(activations) ), ...);
}

template<int N, typename ... T>
void set_initializers(ArtificialNeuralNetwork<N>& ann,
                      std::unique_ptr<T>&& ... initializers)
{
  static_assert( (N-1) == sizeof...(initializers), "number_of_initializers_not_equal_to_number_of_hidden_layers");

  size_t index = 0u;
  ( (ann.initializers[index++] = std::move(initializers) ), ...);

  initialize_params(ann);
}

template<int N>
void initialize_params(ArtificialNeuralNetwork<N>& ann)
{
  int total_weights = 0;
  int total_bias    = 0;
  for (int layer = 1u; layer < N; layer++)
  {
    int n_weights_this_layer = (int)(ann.n_nodes[layer-1u]*ann.n_nodes[layer]);
    int n_bias_this_layer    = (int)(ann.n_nodes[layer]);

    auto this_layer_weights = ann.weights(seq(total_weights, total_weights+n_weights_this_layer-1));
    auto this_layer_bias    = ann.bias(seq(total_bias, total_bias+n_bias_this_layer-1));

    ann.initializers[layer-1u]->initialize((int)ann.n_nodes[layer-1u], this_layer_weights);

    total_weights += n_weights_this_layer;
    total_bias    += n_bias_this_layer;
  }
}


void gradient_clipping(const float threshold, eig::Ref<eig::ArrayXf> gradient)
{
  float gradient_norm = std::sqrtf(gradient.square().sum());
  if (gradient_norm > threshold)
  {
    gradient *= (threshold/gradient_norm);
  }
}

template<int N>
void display_layer_weights_norm(const ANN::ArtificialNeuralNetwork<N>& network)
{
  int weights_start = 0;
  std::vector<float> w_norm(N-1, 0.0F);

  for (int layer = 1u; layer <= N; layer++)
  {
    int n_nodes_last_layer = (int)network.n_nodes[layer-1u];
    int n_nodes_cur_layer  = (int)network.n_nodes[layer];
    int weights_end   = weights_start + (n_nodes_last_layer*n_nodes_cur_layer);

    auto this_layer_w = network.weights(seq(weights_start, weights_end-1));
    w_norm[layer-1] = std::sqrtf(this_layer_w.square().sum());
    weights_start  = weights_end;

    std::cout << " " << layer << ": " << w_norm[layer-1];
  }
  std::cout << '\n';
}

template<int N>
void normalize_if_req(const float norm_threshold, ANN::ArtificialNeuralNetwork<N>& network)
{
  int weights_start = 0;
  for (int layer = 1u; layer <= N; layer++)
  {
    int n_nodes_last_layer = (int)network.n_nodes[layer-1u];
    int n_nodes_cur_layer  = (int)network.n_nodes[layer];
    int weights_end   = weights_start + (n_nodes_last_layer*n_nodes_cur_layer);

    auto this_layer_w = network.weights(seq(weights_start, weights_end-1));
    float this_layer_norm = std::sqrtf(this_layer_w.square().sum());
    if (this_layer_norm > norm_threshold)
    { 
      this_layer_w /= this_layer_norm;
    }

    weights_start  = weights_end;
  }
}

} // namespace {ANN}

#endif
