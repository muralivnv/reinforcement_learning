#ifndef _ANN_FORWARD_H_
#define _ANN_FORWARD_H_

#include "ANN.h"

namespace ann
{

template<int BatchSize, int N, typename EigenDerived>
eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor> 
forward_batch(const ArtificialNeuralNetwork<N>& network, 
              const eig::ArrayBase<EigenDerived>& input)
{
  const size_t largest_layer_len = *std::max_element(network.n_nodes.begin(), network.n_nodes.end());
  constexpr int last_layer_idx = N-1;

  eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor> data_block1(BatchSize, largest_layer_len), 
                                                            data_block2(BatchSize, largest_layer_len), 
                                                            output(BatchSize, (int)network.n_nodes.back());
  int prev_layer_len = (int)input.cols();
  data_block1(all, seq(0, prev_layer_len-1)) = input;

  int weights_count = 0;
  int bias_count    = 0;
  for (int layer = 1u; layer < N; layer++)
  {
    const int n_nodes_last_layer = (int)network.n_nodes[layer-1u];
    const int n_nodes_cur_layer  = (int)network.n_nodes[layer];
    const int& weights_start     = weights_count;
    const int weights_end        = weights_start + n_nodes_last_layer*n_nodes_cur_layer;
    const int& bias_start        = bias_count;
    const int bias_end           = bias_start + n_nodes_cur_layer;
    
    auto prev_layer_activation = data_block1(all, seq(0, prev_layer_len-1));

    auto W    = network.weights(seq(weights_start, weights_end - 1));
    auto B    = network.bias(seq(bias_start, bias_end-1));
    auto WX   = prev_layer_activation.matrix() * W.reshaped(n_nodes_last_layer, n_nodes_cur_layer).matrix();
    auto WX_B = WX.rowwise() + B.matrix().transpose();
    
    if (layer != last_layer_idx)
    { 
      network.activations[layer-1]->activation(WX_B, data_block2.block(0, 0, BatchSize, n_nodes_cur_layer));
      
      // swap data blocks
      data_block1.swap(data_block2);
      prev_layer_len = n_nodes_cur_layer;
      weights_count  = weights_end;
      bias_count     = bias_end;
    }
    else
    {
      network.activations[layer-1]->activation(WX_B, output);
    }
  }

  return output;
}

} // namespace {ann}

#endif
