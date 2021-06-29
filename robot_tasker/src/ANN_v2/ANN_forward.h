#ifndef _ANN_FORWARD_H_
#define _ANN_FORWARD_H_

#include "ANN_typedef.h"
#include "ANN.h"

template<int BatchSize, int N, typename EigenDerived>
eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor> 
forward_batch(const ANN::ArtificialNeuralNetwork<N>& ann, 
              const eig::ArrayBase<EigenDerived>& input)
{
  const size_t largest_layer_len = *std::max_element(ann.n_nodes.begin(), ann.n_nodes.end());

  eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor> data_block1, data_block2;
  int prev_layer_len = input.cols();

  data_block1.resize(BatchSize, largest_layer_len);
  data_block2.resize(BatchSize, largest_layer_len);

  data_block1(all, seq(0, prev_layer_len-1)) = input;

  int weights_count = 0;
  int bias_count    = 0;
  for (int layer = 1u; layer <= N; layer++)
  {
    int n_nodes_last_layer = (int)ann.n_nodes(layer-1u);
    int n_nodes_cur_layer  = (int)ann.n_nodes(layer);
    int weights_start      = weights_count;
    
    auto b = ann.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    auto prev_layer_activation = data_block1(all, seq(0, prev_layer_len-1));

    for (int node = 0; node < n_nodes_cur_layer; node++)
    {
      int this_node_weight_start = weights_start+(node*n_nodes_last_layer);
      int this_node_weight_end   = weights_start+((node+1)*n_nodes_last_layer)-1;
      
      // calculate wX + b
      auto w  = ann.weight(seq(this_node_weight_start, this_node_weight_end));
      auto wx = prev_layer_activation.matrix() * w.matrix();
      auto z = wx.array() + b(node);
      data_block2(all, node) = ann.activations[layer-1].activation_batch(z);
    }

    // swap data blocks
    data_block1.swap(data_block2);
    prev_layer_len = n_nodes_cur_layer;

    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }

  eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor> output (BatchSize, prev_layer_len); 
  output = data_block1(all, seq(0, prev_layer_len-1));
  
  return output;
}


#endif
