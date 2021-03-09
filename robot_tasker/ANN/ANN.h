#ifndef _ANN_H_
#define _ANN_H_

#include <functional>

#include "../global_typedef.h"
#include "ANN_type_traits.h"
#include "ANN_typedef.h"
#include "ANN_activation.h"

template<int InputSize, int ... LayerNodeConfig>
struct ArtificialNeuralNetwork{
  ANN::weights_t     <InputSize, LayerNodeConfig...>    weight;
  ANN::bias_t        <LayerNodeConfig...>               bias;
  ANN::activations_t <LayerNodeConfig...>               layer_activation_func;
  ANN::layerConfig_t <InputSize, LayerNodeConfig...>    n_nodes;
  size_t                                                n_layers;

  ANN::lossFcn_t          loss;
  ANN::lossFcnGrad_t      loss_grad;

  ArtificialNeuralNetwork()
  { this->set_layer_config(InputSize, LayerNodeConfig...); }

  ~ArtificialNeuralNetwork(){}

  template<typename ... LayerConfig>
  void set_layer_config(LayerConfig... layer_config)
  {
    size_t index = 0u;
    ((this->n_nodes[index++] = layer_config), ...);
    this->n_layers = index - 1u;
  }

  template<typename ... Activation_t>
  void dense(Activation_t& ... activation)
  {
    size_t index = 0u;
    ((this->layer_activation_func[index++] = activation), ...);
    init_params();
  }

  void init_params()
  {
    int total_weights = 0u;
    int total_bias    = 0u;
    for (int layer = 1u; layer < (int)this->n_layers; layer++)
    {
      int n_weights_this_layer = (int)(this->n_nodes[layer-1u]*this->n_nodes[layer]);
      int n_bias_this_layer    = (int)(this->n_nodes[layer]);

      auto this_layer_weights = this->weight(seq(total_weights, total_weights+n_weights_this_layer));
      auto this_layer_bias    = this->bias(seq(total_bias, total_bias+n_bias_this_layer));

      this->layer_activation_func[layer-1u].initializer.init(this_layer_weights, this_layer_bias, (int)this->n_nodes[layer-1u]);

      total_weights += n_weights_this_layer;
      total_bias    += n_bias_this_layer;
    }
  }
};


template<int InputSize, int ... LayerNodeConfig, typename EigenDerived>
auto forward(const ArtificialNeuralNetwork<InputSize, LayerNodeConfig...>& ann, const eig::DenseBase<EigenDerived>& input)     
  -> ANN::output_t<LayerNodeConfig...>
{
  RL::VectorX<float> last_activation = input(seq(0, InputSize-1u));

  size_t weights_count = 0u;
  size_t bias_count    = 0u;
  for (size_t layer = 1u; layer < ann.n_layers; layer++)
  {
    size_t n_nodes_last_layer = ann.n_nodes(layer-1u);
    size_t n_nodes_cur_layer  = ann.n_nodes(layer);
    size_t weights_start      = weights_count;
    size_t weights_end        = weights_start + (n_nodes_cur_layer*n_nodes_last_layer);
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1u));
    RL::VectorX<float> this_layer_activation(n_nodes_cur_layer);
    for (size_t node = 0u; node < n_nodes_cur_layer; node++)
    {
      // calculate wX + b
      auto wx = last_activation.dot(ann.weight(seq(weights_start+(node*n_nodes_last_layer), weights_start+((node+1)*n_nodes_last_layer)-1)));
      this_layer_activation(node) = (ann.layer_activation_func[layer-1]).activation(wx + b(node));
    }
    last_activation = std::move(this_layer_activation);
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }

  ANN::output_t<LayerNodeConfig...> output = std::move(last_activation);
  return output;
}


template<int BatchSize, int InputSize, int ... LayerNodeConfig, typename EigenDerived>
auto forward_batch(const ArtificialNeuralNetwork<InputSize, LayerNodeConfig...>& ann, const eig::DenseBase<EigenDerived>& input)
        -> ANN::output_batch_t<BatchSize, LayerNodeConfig...>
{
  RL::MatrixX<float> last_activation = input(all, seq(0, InputSize-1u));

  size_t weights_count = 0u;
  size_t bias_count    = 0u;
  for (size_t layer = 1u; layer < ann.n_layers; layer++)
  {
    size_t n_nodes_last_layer = ann.n_nodes(layer-1u);
    size_t n_nodes_cur_layer  = ann.n_nodes(layer);
    size_t weights_start      = weights_count;
    size_t weights_end        = weights_start + (n_nodes_cur_layer*n_nodes_last_layer);
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1u));
    RL::MatrixX<float> this_layer_activation(BatchSize, n_nodes_cur_layer);

    for (size_t node = 0u; node < n_nodes_cur_layer; node++)
    {
      // calculate wX + b
      auto wx = last_activation * ann.weight(seq(weights_start+(node*n_nodes_last_layer), weights_start+((node+1)*n_nodes_last_layer)-1));
      this_layer_activation(all, bias_count+node) = ann.layer_activation_func[layer-1].activation_batch(wx.array() + b(node));
    }
    last_activation = this_layer_activation;
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }
  ANN::output_batch_t<BatchSize, LayerNodeConfig...> output = last_activation;
  return output;
}


template<int BatchSize, int InputSize, int ... LayerNodeConfig, typename EigenDerived1, typename EigenDerived2>
auto gradient_batch(const ArtificialNeuralNetwork<InputSize, LayerNodeConfig...>&  ann, 
                    const eig::DenseBase<EigenDerived1>&                           input,
                    const eig::DenseBase<EigenDerived2>&                           ref_out)
{
  constexpr size_t output_len = ann_output_len<LayerNodeConfig...>::value;
  constexpr size_t n_layers   = pack_len<InputSize, LayerNodeConfig...>::value;

  // Perform forward propagation
  vector<RL::MatrixX<float>> activations(n_layers);
  tuple<decltype(ann.weight), decltype(ann.bias)> retval = std::make_tuple(decltype(ann.weight){}, decltype(ann.bias){});
  auto& [weight_grad, bias_grad] = retval;
  
  activations[0] = RL::MatrixX<float>(input.rows(), input.cols());
  activations[0] = input;

  size_t weights_count = 0u;
  size_t bias_count    = 0u;
  for (size_t layer = 1u; layer < ann.n_layers; layer++)
  {
    size_t n_nodes_last_layer = ann.n_nodes(layer-1u);
    size_t n_nodes_cur_layer  = ann.n_nodes(layer);
    size_t weights_start      = weights_count;
    size_t weights_end        = weights_start + (n_nodes_cur_layer*n_nodes_last_layer);
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1u));
    activations[layer] = RL::MatrixX<float>(BatchSize, n_nodes_cur_layer);
    for (size_t node = 0u; node < n_nodes_cur_layer; node++)
    {
      // calculate wX + b
      auto wx = activations[layer-1] * ann.weight(seq(weights_start+node, weights_end, n_nodes_cur_layer));
      activations[layer](all, node) = ann.layer_activation_func[layer-1].activation_batch(wx.array() + b(node));
    }
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }
  
  // perform backward propagation to calculate gradient
  auto loss_grad = ann.loss_grad(activations.back(), ref_out);

  // TODO: Write backward propagation

  return std::make_tuple(weight_grad, bias_grad);
}

#endif
