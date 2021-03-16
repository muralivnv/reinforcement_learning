#ifndef _ANN_OPTIMIZERS_H_
#define _ANN_OPTIMIZERS_H_

#include <random>
#include "../global_typedef.h"

#include "ANN_type_traits.h"
#include "ANN_typedef.h"
#include "ANN.h"

template<int BatchSize, int InputSize, int ...LayerNodeCondfig, typename EigenDerived1, typename EigenDerived2>
void steepest_descent(const eig::DenseBase<EigenDerived1>& input, 
                      const eig::DenseBase<EigenDerived2>& ref_out,
                      const ANN::OptimizerParams&          params,
                            ArtificialNeuralNetwork<InputSize, NHiddenLayers...>& ann)
{
  static const size_t n_epochs    = std::any_cast<size_t>(params.at("n_epochs"));
  static const float step_size    = std::any_cast<float>(params.at("step_size"));

  for (size_t epoch = 0u; epoch < n_epochs; epoch++)
  {
    // calculate gradient of weight and bias
    auto [weight_grad, bias_grad] = gradient_batch(ann, input, ref_out);

    // update weights
    ann.weight -= (step_size*weight_grad); 
    
    // update bias
    ann.bias   -= (step_size*bias_grad);
  }
}

#endif
