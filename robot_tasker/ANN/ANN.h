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
  void dense(Activation_t&& ... activation)
  {
    size_t index = 0u;
    ((this->layer_activation_func[index++] = activation), ...);
    init_params();
  }

  void init_params()
  {
    int total_weights = 0;
    int total_bias    = 0;
    for (int layer = 1u; layer <= (int)this->n_layers; layer++)
    {
      int n_weights_this_layer = (int)(this->n_nodes[layer-1u]*this->n_nodes[layer]);
      int n_bias_this_layer    = (int)(this->n_nodes[layer]);

      auto this_layer_weights = this->weight(seq(total_weights, total_weights+n_weights_this_layer-1));
      auto this_layer_bias    = this->bias(seq(total_bias, total_bias+n_bias_this_layer-1));

      this->layer_activation_func[layer-1u].initializer.init(this_layer_weights, this_layer_bias, (int)this->n_nodes[layer-1u]);

      total_weights += n_weights_this_layer;
      total_bias    += n_bias_this_layer;
    }
  }
};


template<int InputSize, int ... LayerNodeConfig, typename EigenDerived>
auto forward(const ArtificialNeuralNetwork<InputSize, LayerNodeConfig...>& ann, const eig::ArrayBase<EigenDerived>& input)     
  -> ANN::output_t<LayerNodeConfig...>
{
  constexpr int largest_layer_len = max_layer_len<InputSize, LayerNodeConfig...>::value;
  eig::Array<float, eig::Dynamic, eig::Dynamic, 0, largest_layer_len, 1> prev_layer_activation;
  eig::Array<float, eig::Dynamic, eig::Dynamic, 0, largest_layer_len, 1> this_layer_activation;

  prev_layer_activation.resize(input.rows(), input.cols());
  prev_layer_activation = input;

  size_t weights_count = 0u;
  size_t bias_count    = 0u;
  for (size_t layer = 1u; layer < ann.n_layers; layer++)
  {
    size_t n_nodes_last_layer = ann.n_nodes(layer-1u);
    size_t n_nodes_cur_layer  = ann.n_nodes(layer);
    size_t weights_start      = weights_count;
    size_t weights_end        = weights_start + (n_nodes_cur_layer*n_nodes_last_layer);
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1u));
    this_layer_activation.resize(n_nodes_cur_layer, 1);
    for (size_t node = 0u; node < n_nodes_cur_layer; node++)
    {
      // calculate wX + b
      auto wx = prev_layer_activation.dot( ann.weight(seq(weights_start+(node*n_nodes_last_layer), weights_start+((node+1)*n_nodes_last_layer)-1)) );
      this_layer_activation(node) = (ann.layer_activation_func[layer-1]).activation(wx + b(node));
    }
    prev_layer_activation.swap(this_layer_activation);
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }

  ANN::output_t<LayerNodeConfig...> output = prev_layer_activation;
  return output;
}


template<int BatchSize, int InputSize, int ... LayerNodeConfig, typename EigenDerived>
auto forward_batch(const ArtificialNeuralNetwork<InputSize, LayerNodeConfig...>& ann, const eig::ArrayBase<EigenDerived>& input)
        -> ANN::output_batch_t<BatchSize, LayerNodeConfig...>
{
  constexpr int largest_layer_len = max_layer_len<InputSize, LayerNodeConfig...>::value;
  eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor, BatchSize, largest_layer_len> prev_layer_activation;
  eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor, BatchSize, largest_layer_len> this_layer_activation;
  prev_layer_activation.resize(input.rows(), input.cols());
  prev_layer_activation = input;

  int weights_count = 0;
  int bias_count    = 0;
  for (int layer = 1u; layer <= (int)ann.n_layers; layer++)
  {
    int n_nodes_last_layer = (int)ann.n_nodes(layer-1u);
    int n_nodes_cur_layer  = (int)ann.n_nodes(layer);
    int weights_start      = weights_count;
    int weights_end        = weights_start + (n_nodes_cur_layer*n_nodes_last_layer) - 1;
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    this_layer_activation.resize(BatchSize, n_nodes_cur_layer);

    for (size_t node = 0; node < n_nodes_cur_layer; node++)
    {
      // calculate wX + b
      auto w  = ann.weight(seq(weights_start+node, weights_end, n_nodes_cur_layer)).eval();
      auto wx = prev_layer_activation.matrix() * w.matrix();
      this_layer_activation(all, node) = ann.layer_activation_func[layer-1].activation_batch(wx.array() + b(node));
    }
    prev_layer_activation.swap(this_layer_activation);
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }
  ANN::output_batch_t<BatchSize, LayerNodeConfig...> output = prev_layer_activation;
  return output;
}


template<int BatchSize, int InputSize, int ... LayerNodeConfig, typename EigenDerived1, typename EigenDerived2, 
         typename LossFcn_t, typename LossGradFcn_t>
auto gradient_batch(const ArtificialNeuralNetwork<InputSize, LayerNodeConfig...>&  ann, 
                    const eig::ArrayBase<EigenDerived1>&                           input,
                    const eig::ArrayBase<EigenDerived2>&                           ref_out, 
                          LossFcn_t&                                               loss_fcn,
                          LossGradFcn_t&                                           loss_grad_fcn)
{
  constexpr int output_len = ann_output_len<LayerNodeConfig...>::value;
  constexpr int n_layers   = pack_len<InputSize, LayerNodeConfig...>::value;

  // Perform forward propagation
  vector<eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>> activations(n_layers);
  tuple<decltype(ann.weight), decltype(ann.bias)> retval = std::make_tuple(decltype(ann.weight){}, decltype(ann.bias){});
  auto& [weight_grad, bias_grad] = retval;
  
  activations[0] = eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>(input.rows(), input.cols());
  activations[0] = input;

  int weights_count = 0;
  int bias_count    = 0;
  for (int layer = 1; layer <= (int)ann.n_layers; layer++)
  {
    int n_nodes_last_layer = (int)ann.n_nodes(layer-1);
    int n_nodes_cur_layer  = (int)ann.n_nodes(layer);
    int weights_start      = weights_count;
    int weights_end        = weights_start + (n_nodes_cur_layer*n_nodes_last_layer) - 1;
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    activations[layer] = eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>(BatchSize, n_nodes_cur_layer);
    for (int node = 0; node < n_nodes_cur_layer; node++)
    {
      // calculate wX + b
      auto w  = ann.weight(seq(weights_start+node, weights_end, n_nodes_cur_layer)).eval();
      auto wx = (activations[layer-1].matrix() * w.matrix());
      activations[layer](all, node) = ann.layer_activation_func[layer-1].activation_batch( (wx.array() + b(node)) );
    }
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }
  
  // perform backward propagation to calculate gradient
  auto loss_grad = loss_grad_fcn(activations.back(), ref_out);

  // calculate delta for last layer
  constexpr int largest_layer_len = max_layer_len<InputSize, LayerNodeConfig...>::value;
  eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor, BatchSize, largest_layer_len> delta;
  eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor, BatchSize, largest_layer_len> delta_to_here;

  int n_nodes_this_layer = (int)ann.n_nodes(last);
  int n_nodes_prev_layer = (int)ann.n_nodes(last-1);

  decltype(delta) this_layer_activation_grad = ann.layer_activation_func.back().grad(activations.back());
  delta.resize(loss_grad.rows(), this_layer_activation_grad.cols());
  delta_to_here.resize(delta.rows(), delta.cols());

  decltype(delta) prev_layer_activations = activations[ann.n_layers-1];
  delta = loss_grad * this_layer_activation_grad;
  
  int weight_end   = (int)ann.weight.rows()-1;
  int weight_start = weight_end - (n_nodes_this_layer*n_nodes_prev_layer) + 1;
  int bias_end     = (int)ann.bias.rows()-1;
  int bias_start   = bias_end   - (n_nodes_this_layer);

  for (int node = 0; node < n_nodes_this_layer; node++)
  {
    for (int j = 0; j < n_nodes_prev_layer; j++)
    {
      auto temp = delta(all, node)*prev_layer_activations(all, j);
      weight_grad(weight_start+node*n_nodes_prev_layer+j) = temp.mean();
    }
    bias_grad(bias_start+node) = delta(all, node).mean();
  }

  // calculate gradient for remaining layers
  delta_to_here.swap(delta);
  int next_layer_weights_end   = weight_end;
  int next_layer_weights_start = weight_start;
  for (int layer = (int)ann.n_layers-1; layer > 0; layer--)
  {
    n_nodes_this_layer         = ann.n_nodes[layer];
    n_nodes_prev_layer         = ann.n_nodes[layer-1];
    int n_nodes_next_layer     = ann.n_nodes[layer+1];
    this_layer_activation_grad = ann.layer_activation_func[layer].grad(activations[layer]);
    prev_layer_activations     = activations[layer-1];

    delta.resize(BatchSize, n_nodes_this_layer);

    // calculate delta
    for (int node = 0; node < n_nodes_this_layer; node++)
    {
      delta(all, node) = delta_to_here.matrix()*ann.weight(seq(next_layer_weights_start+node, next_layer_weights_end, n_nodes_this_layer)).eval().matrix();
    }
    delta *= this_layer_activation_grad;

    // calculate gradient
    weight_end   = next_layer_weights_start - 1;
    weight_start = weight_end - (n_nodes_this_layer*n_nodes_prev_layer) + 1;
    for (int node = 0; node < n_nodes_this_layer; node++)
   {
      for (int j = 0; j < n_nodes_prev_layer; j++)
      {
        auto temp = delta(all, node)*prev_layer_activations(all, j);
        weight_grad(weight_start+(node*n_nodes_prev_layer)+j) = temp.mean();
      }
      bias_grad(bias_start+node) = delta(all, node).mean();
   }

    delta_to_here.swap(delta);

    next_layer_weights_end   = weight_end;
    next_layer_weights_start = weight_start;
  }

  return retval;
}

#endif
