#include <iostream>

#include "global_typedef.h"
#include "environment_util.h"
#include "ANN/ANN.h"
#include "cppyplot.hpp"

using namespace ANN;

int main()
{
  vector<RL::Polynomial<1>> world = ENV::read_world("global_config.yaml");

  ArtificialNeuralNetwork<2, 3, 3, 2> sampling_policy;
  sampling_policy.dense(ReLU(Initializer(HE)), 
                        ReLU(Initializer(HE)), 
                        ReLU(Initializer(HE)) );
  
  RL::Matrix<float, 25, 2> X;
  RL::Matrix<float, 25, 2> Y;
  auto out = gradient_batch<25>(sampling_policy, X, Y);

  // TODO: Make it compile
  // TODO: Test ANN with a sample test
  return EXIT_SUCCESS;
}