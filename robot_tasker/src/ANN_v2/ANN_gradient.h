#ifndef _ANN_GRADIENT_H_
#define _ANN_GRADIENT_H_

#include "ANN_typedef.h"
#include "ANN.h"

#include "ANN_activation.h"

template<int BatchSize, int N, typename EigenDerived1, typename EigenDerived2, 
         typename LossFcn, typename LossGradFcn>
tuple<float, ANN::weights_t, ANN::bias_t>
gradient_batch(const ANN::ArtificialNeuralNetwork<N>&    ann, 
               const eig::ArrayBase<EigenDerived1>& input,
               const eig::ArrayBase<EigenDerived2>& ref_out,
                     LossFcn&                       loss_fcn, 
                     LossGradFcn&                   loss_grad_fcn)
{
  const int output_len        = (int)ann.n_nodes.back();
  const int n_layers          = N;
  const int largest_layer_len = (int)*std::max_element(ann.n_nodes.begin(), ann.n_nodes.end());
  using weight_type           = decltype(ann.weights);
  using bias_type             = decltype(ann.bias);
  
  using Delta_t          = eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor>;
  using Activation_t     = eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor>;

  tuple<float, weight_type, bias_type> retval = std::make_tuple(0.0F, weight_type{}, bias_type{});
  auto& [loss, weight_grad, bias_grad] = retval;

  // Perform forward propagation
  vector<Activation_t> activations; 
  activations.reserve(N);
  activations.emplace_back(input);

  int weights_count = 0;
  int bias_count    = 0;
  for (int layer = 1; layer <= N; layer++)
  {
    int n_nodes_last_layer = (int)ann.n_nodes[layer-1];
    int n_nodes_cur_layer  = (int)ann.n_nodes[layer];
    int weights_start      = weights_count;
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    activations.emplace_back(BatchSize, n_nodes_cur_layer);
    auto& this_activation = activations[layer];
    for (int node = 0; node < n_nodes_cur_layer; node++)
    {
      int this_node_weight_start = weights_start+(node*n_nodes_last_layer);
      int this_node_weight_end   = weights_start+((node+1)*n_nodes_last_layer)-1;

      // calculate wX + b
      auto w  = ann.weights(seq(this_node_weight_start, this_node_weight_end));
      auto wx = (activations[layer-1].matrix() * w.matrix());
      auto z = wx.array() + b(node);
      ann.activations[layer-1]->activation(z, this_activation(all, node));
    }
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }
  // calculate loss
  loss = loss_fcn(activations.back(), ref_out);

  // perform backward propagation to calculate gradient
  // calculate delta for last layer
  Delta_t delta(BatchSize, largest_layer_len), delta_to_here(BatchSize, largest_layer_len);
  delta_to_here(all, seq(0, ann.n_nodes.back()-1)) = loss_grad_fcn(activations.back(), ref_out);
  
  int prev_delta_n_cols = (int)ann.n_nodes.back();
  int n_nodes_this_layer, n_nodes_prev_layer;
  int weight_end, weight_start;
  int bias_end, bias_start;

  int next_layer_weights_end = 0, next_layer_weights_start = (int)ann.weights.rows();
  int next_layer_bias_end = 0, next_layer_bias_start = (int)ann.bias.rows();
  for (int layer = N; layer != 0; layer--)
  {
    n_nodes_this_layer              = (int)ann.n_nodes[layer];
    n_nodes_prev_layer              = (int)ann.n_nodes[layer-1];
    auto& prev_layer_activations    = activations[layer-1];
    Activation_t this_layer_activation_grad;
    ann.activations[layer-1]->grad(activations[layer], this_layer_activation_grad);

    // calculate delta
    if (layer != N)
    {
      auto delta_before = delta_to_here(all, seq(0, prev_delta_n_cols-1));
      for (int node = 0; node < n_nodes_this_layer; node++)
      {
        delta(all, node) = (delta_to_here.matrix()*ann.weights(seq(next_layer_weights_start+node, 
                                                                   next_layer_weights_end, 
                                                                   n_nodes_this_layer)).eval().matrix());
      }
    }
    else 
    { delta.swap(delta_to_here); }

    delta(all, seq(0, n_nodes_this_layer-1)) *= this_layer_activation_grad;

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
    prev_delta_n_cols        = n_nodes_this_layer;
  }
  return retval;
}

#endif
