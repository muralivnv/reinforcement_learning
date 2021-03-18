#ifndef _ANN_OPTIMIZERS_H_
#define _ANN_OPTIMIZERS_H_

#include <random>
#include "../global_typedef.h"

#include "ANN_type_traits.h"
#include "ANN_typedef.h"
#include "ANN.h"

template<int BatchSize, int InputSize, int ...LayerNodeCondfig, typename EigenDerived1, typename EigenDerived2>
void steepest_descent(const eig::ArrayBase<EigenDerived1>& weight_grad, 
                      const eig::ArrayBase<EigenDerived2>& bias_grad,
                      const ANN::OptimizerParams&          params,
                            ArtificialNeuralNetwork<InputSize, NHiddenLayers...>& network)
{
  static const float& step_size    = std::any_cast<float>(params.at("step_size"));

    // update weights
    network.weight -= (step_size*weight_grad); 
    
    // update bias
    network.bias   -= (step_size*bias_grad);
  }
}

#endif
