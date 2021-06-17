#ifndef _ANN_H_
#define _ANN_H_

#include <functional>

#include "../global_typedef.h"
#include "ANN_type_traits.h"
#include "ANN_typedef.h"
#include "ANN_activation.h"

namespace ANN
{

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
    static_assert(sizeof...(activation) == sizeof...(NHiddenLayers), "number_of_activations_not_equal_to_number_of_hidden_layers");
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
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    this_layer_activation.conservativeResize(NoChange, n_nodes_cur_layer);

    for (int node = 0; node < n_nodes_cur_layer; node++)
    {
      int this_node_weight_start = weights_start+(node*n_nodes_last_layer);
      int this_node_weight_end   = weights_start+((node+1)*n_nodes_last_layer)-1;
      
      // calculate wX + b
      auto w  = ann.weight(seq(this_node_weight_start, this_node_weight_end));
      auto wx = prev_layer_activation.matrix() * w.matrix();
      auto z = wx.array() + b(node);
      this_layer_activation(all, node) = ann.layer_activation_func[layer-1].activation_batch(z);
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
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    activations.emplace_back(BatchSize, n_nodes_cur_layer);
    for (int node = 0; node < n_nodes_cur_layer; node++)
    {
      int this_node_weight_start = weights_start+(node*n_nodes_last_layer);
      int this_node_weight_end   = weights_start+((node+1)*n_nodes_last_layer)-1;

      // calculate wX + b
      auto w  = ann.weight(seq(this_node_weight_start, this_node_weight_end));
      auto wx = (activations[layer-1].matrix() * w.matrix());
      auto z = wx.array() + b(node);
      activations[layer](all, node) = ann.layer_activation_func[layer-1].activation_batch( z );
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

template<int BatchSize, int InputSize, int ... NHiddenLayers, typename EigenDerived1, 
         typename LossFcn_t, typename LossGradFcn_t>
auto gradient_batch(const ArtificialNeuralNetwork<InputSize, NHiddenLayers...>&  ann, 
                    const eig::ArrayBase<EigenDerived1>&                         input,
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
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    activations.emplace_back(BatchSize, n_nodes_cur_layer);
    for (int node = 0; node < n_nodes_cur_layer; node++)
    {
      int this_node_weight_start = weights_start+(node*n_nodes_last_layer);
      int this_node_weight_end   = weights_start+((node+1)*n_nodes_last_layer)-1;

      // calculate wX + b
      auto w  = ann.weight(seq(this_node_weight_start, this_node_weight_end));
      auto wx = (activations[layer-1].matrix() * w.matrix());
      auto z = wx.array() + b(node);
      activations[layer](all, node) = ann.layer_activation_func[layer-1].activation_batch( z );
    }
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }
  // calculate loss
  loss = loss_fcn(activations.back());

  // perform backward propagation to calculate gradient
  // calculate delta for last layer
  Delta_t delta, delta_to_here(loss_grad_fcn(activations.back()));
  
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


} //namespace {ANN}
#endif
