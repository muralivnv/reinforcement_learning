#ifndef _ANN_H_
#define _ANN_H_

#include <functional>

#include "../global_typedef.h"
#include "ANN_type_traits.h"
#include "ANN_typedef.h"
#include "ANN_activation.h"

template<int InputSize, int ... NHiddenLayers>
struct ArtificialNeuralNetwork{
  ANN::weights_t     <InputSize, NHiddenLayers...>    weight;
  ANN::bias_t        <NHiddenLayers...>               bias;
  ANN::activations_t <NHiddenLayers...>               layer_activation_func;
  ANN::layerConfig_t <InputSize, NHiddenLayers...>    n_nodes;
  int                                                 n_layers;

  ArtificialNeuralNetwork()
  { this->set_layer_config(InputSize, NHiddenLayers...); }

  ~ArtificialNeuralNetwork(){}

  template<typename ... LayerConfig>
  void set_layer_config(LayerConfig... layer_config)
  {
    int index = 0;
    ((this->n_nodes[index++] = layer_config), ...);
    this->n_layers = index - 1;
  }

  template<typename ... Activation_t>
  void dense(Activation_t&& ... activation)
  {
    int index = 0u;
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


template<int InputSize, int ... NHiddenLayers, typename EigenDerived>
auto forward(const ArtificialNeuralNetwork<InputSize, NHiddenLayers...>& ann, const eig::ArrayBase<EigenDerived>& input)     
  -> ANN::output_t<NHiddenLayers...>
{
  constexpr int largest_layer_len = max_layer_len<InputSize, NHiddenLayers...>::value;
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

  ANN::output_t<NHiddenLayers...> output = prev_layer_activation;
  return output;
}


template<int BatchSize, int InputSize, int ... NHiddenLayers, typename EigenDerived>
auto forward_batch(const ArtificialNeuralNetwork<InputSize, NHiddenLayers...>& ann, const eig::ArrayBase<EigenDerived>& input)
        -> ANN::output_batch_t<BatchSize, NHiddenLayers...>
{
  constexpr int largest_layer_len = max_layer_len<InputSize, NHiddenLayers...>::value;
  eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor, BatchSize, largest_layer_len> prev_layer_activation;
  eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor, BatchSize, largest_layer_len> this_layer_activation;

  prev_layer_activation.conservativeResize(NoChange, input.cols());
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
    this_layer_activation.conservativeResize(NoChange, n_nodes_cur_layer);

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
  ANN::output_batch_t<BatchSize, NHiddenLayers...> output = prev_layer_activation;
  return output;
}


template<int BatchSize, int InputSize, int ... NHiddenLayers, typename EigenDerived1, typename EigenDerived2, 
         typename LossFcn_t, typename LossGradFcn_t>
auto gradient_batch(const ArtificialNeuralNetwork<InputSize, NHiddenLayers...>&  ann, 
                    const eig::ArrayBase<EigenDerived1>&                         input,
                    const eig::ArrayBase<EigenDerived2>&                         ref_out,
                          LossFcn_t&                                             loss_fcn, 
                          LossGradFcn_t&                                         loss_grad_fcn)
{
  constexpr int output_len        = ann_output_len<NHiddenLayers...>::value;
  constexpr int n_layers          = pack_len<InputSize, NHiddenLayers...>::value;
  constexpr int largest_layer_len = max_layer_len<InputSize, NHiddenLayers...>::value;
  using weight_type      = decltype(ann.weight);
  using bias_type        = decltype(ann.bias);
  using Delta_t          = eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor, BatchSize, largest_layer_len>;
  using Activation_t     = eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>;

  tuple<float, weight_type, bias_type> retval = std::make_tuple(0.0F, weight_type{}, bias_type{});
  auto& [loss, weight_grad, bias_grad] = retval;

  // Perform forward propagation
  vector<Activation_t> activations; 
  activations.reserve(n_layers);
  activations.emplace_back(input);

  int weights_count = 0;
  int bias_count    = 0;
  for (int layer = 1; layer <= ann.n_layers; layer++)
  {
    int n_nodes_last_layer = ann.n_nodes(layer-1);
    int n_nodes_cur_layer  = ann.n_nodes(layer);
    int weights_start      = weights_count;
    int weights_end        = weights_start + (n_nodes_cur_layer*n_nodes_last_layer) - 1;
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    activations.emplace_back(BatchSize, n_nodes_cur_layer);
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
  // calculate loss
  loss = loss_fcn(activations.back(), ref_out);

  // perform backward propagation to calculate gradient
  // calculate delta for last layer
  Delta_t delta, delta_to_here(loss_grad_fcn(activations.back(), ref_out));
  
  int n_nodes_this_layer, n_nodes_prev_layer;
  int weight_end, weight_start;
  int bias_end, bias_start;

  int next_layer_weights_end = 0, next_layer_weights_start = (int)ann.weight.rows();
  int next_layer_bias_end = 0, next_layer_bias_start = (int)ann.bias.rows();
  for (int layer = ann.n_layers; layer != 0; layer--)
  {
    n_nodes_this_layer              = ann.n_nodes[layer];
    n_nodes_prev_layer              = ann.n_nodes[layer-1];
    auto this_layer_activation_grad = ann.layer_activation_func[layer-1].grad(activations[layer]);
    auto& prev_layer_activations    = activations[layer-1];

    // calculate delta
    if (layer != ann.n_layers)
    {
      delta.conservativeResize(NoChange, n_nodes_this_layer);
      for (int node = 0; node < n_nodes_this_layer; node++)
      {
        delta(all, node) = (delta_to_here.matrix()*ann.weight(seq(next_layer_weights_start+node, 
                                                                  next_layer_weights_end, 
                                                                  n_nodes_this_layer)).eval().matrix());
      }
    }
    else 
    { delta.swap(delta_to_here); }
    delta *= this_layer_activation_grad;

    // calculate gradient
    weight_end   = next_layer_weights_start - 1;
    weight_start = weight_end - (n_nodes_this_layer*n_nodes_prev_layer) + 1;
    bias_end     = next_layer_bias_start - 1;
    bias_start   = bias_end - n_nodes_this_layer + 1;
   for (int node = 0; node < n_nodes_this_layer; node++)
   {
      auto temp = delta(all, node);
      for (int j = 0; j < n_nodes_prev_layer; j++)
      {
        auto weight_grad_list = temp*prev_layer_activations(all, j);
        weight_grad(weight_start+(node*n_nodes_prev_layer)+j) = weight_grad_list.mean();
      }
      bias_grad(bias_start+node) = temp.mean();
   }
    delta_to_here.swap(delta);

    next_layer_weights_end   = weight_end;
    next_layer_weights_start = weight_start;
    next_layer_bias_start    = bias_start;
    next_layer_bias_end      = bias_end;
  }
  return retval;
}

#endif
