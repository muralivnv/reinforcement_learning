#ifndef _ANN_GRADIENT_H_
#define _ANN_GRADIENT_H_

#include "ANN.h"
#include "ANN_activation.h"

namespace ann
{

template<int BatchSize, int N, typename EigenDerived1, typename EigenDerived2, 
         typename LossFcn, typename LossGradFcn>
tuple<float, weights_t, bias_t>
gradient_batch(const ArtificialNeuralNetwork<N>&    network, 
               const eig::ArrayBase<EigenDerived1>& input,
               const eig::ArrayBase<EigenDerived2>& ref_out,
                     LossFcn&                       loss_fcn, 
                     LossGradFcn&                   loss_grad_fcn)
{
  const int output_len        = (int)network.n_nodes.back();
  const int largest_layer_len = (int)*std::max_element(network.n_nodes.begin(), network.n_nodes.end());
  
  using Delta_t          = eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor>;
  using Activation_t     = eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor>;

  tuple<float, weights_t, bias_t> retval = std::make_tuple(0.0F, weights_t{}, bias_t{});
  auto& [loss, weight_grad, bias_grad] = retval;
  weight_grad.resize(network.weights.rows(), 1);
  bias_grad.resize(network.bias.rows(), 1);

  // Perform forward propagation
  vector<Activation_t> activations;
  vector<Activation_t>& delta = activations;
  activations.reserve(N);
  activations.emplace_back(input);

  int weights_count = 0;
  int bias_count    = 0;
  for (int layer = 1; layer < N; layer++)
  {
    const int n_nodes_last_layer = (int)network.n_nodes[layer-1];
    const int n_nodes_cur_layer  = (int)network.n_nodes[layer];
    const int& weights_start     = weights_count;
    const int weights_end        = weights_start + n_nodes_cur_layer*n_nodes_last_layer;
    const int& bias_start        = bias_count;
    const int bias_end           = bias_start + n_nodes_cur_layer;
    
    activations.emplace_back(BatchSize, n_nodes_cur_layer);
    auto& this_activation = activations[layer];

    auto W    = network.weights(seq(weights_start, weights_end - 1));
    auto B    = network.bias(seq(bias_start, bias_end-1));
    auto WX   = activations[layer-1].matrix() * W.reshaped(n_nodes_last_layer, n_nodes_cur_layer).matrix();
    auto WX_B = WX.rowwise() + B.matrix().transpose();
    network.activations[layer-1]->activation(WX_B, this_activation);

    weights_count  = weights_end;
    bias_count     = bias_end;
  }

  // calculate loss
  loss = loss_fcn(activations.back(), ref_out);

  // perform backward propagation to calculate gradient
  Activation_t this_layer_activation_grad(BatchSize, largest_layer_len);

  int n_nodes_this_layer, n_nodes_prev_layer;
  int weight_end, weight_start;
  int bias_end, bias_start;

  int next_layer_weights_end = 0, next_layer_weights_start = (int)network.weights.rows();
  int next_layer_bias_start = (int)network.bias.rows();
  for (int layer = N-1; layer != 0; layer--)
  {
    n_nodes_this_layer              = (int)network.n_nodes[layer];
    n_nodes_prev_layer              = (int)network.n_nodes[layer-1];
    auto& prev_layer_activations    = activations[layer-1];
    auto this_layer_grad            = this_layer_activation_grad.block(0, 0, BatchSize, n_nodes_this_layer);
    network.activations[layer-1]->grad(activations[layer], this_layer_grad);
    
    auto& delta_now = delta[layer];

    // calculate delta
    if (layer != (N-1))
    {
      auto& delta_next   = delta[layer+1];
      auto W_trans       = (network.weights(seq(next_layer_weights_start, next_layer_weights_end))).matrix().transpose();
      delta_now          = delta_next.matrix() * W_trans;
    }
    else 
    {
      delta_now = loss_grad_fcn(activations.back(), ref_out);
    }
    delta_now *= this_layer_grad;

    // calculate gradient
    weight_end   = next_layer_weights_start - 1;
    weight_start = weight_end - (n_nodes_this_layer*n_nodes_prev_layer) + 1;
    bias_end     = next_layer_bias_start - 1;
    bias_start   = bias_end - n_nodes_this_layer + 1;
   for (int node = 0; node < n_nodes_this_layer; node++)
   {
      auto temp = delta_now(all, node);
      int n = weight_start + (node*n_nodes_prev_layer);
      int m = n + n_nodes_prev_layer-1;

      weight_grad(seq(n, m)) = (prev_layer_activations.colwise() * temp).colwise().mean();
      bias_grad(bias_start+node) = temp.mean();
   }

    next_layer_weights_end   = weight_end;
    next_layer_weights_start = weight_start;
    next_layer_bias_start    = bias_start;
  }
  return retval;
}

} // namespace {ann}

#endif
