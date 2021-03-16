#include <iostream>
#include <random>

#include "global_typedef.h"
#include "environment_util.h"
#include "ANN/ANN.h"
#include "cppyplot.hpp"

using namespace ANN;
#define WORLD_FILE ("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/world_barriers.csv")

// template<typename EigenDerived1, typename EigenDerived2>
eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>
loss(const eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>& pred_out, 
     const eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>& ref_out)
{
  eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor> 
  retval (pred_out.rows(), 1);
  retval(all, 0)  = ((pred_out(all, 0) - ref_out(all, 0))*(pred_out(all, 0) - ref_out(all, 0)));
  retval(all, 0) += ((pred_out(all, 1) - ref_out(all, 1))*(pred_out(all, 1) - ref_out(all, 1)));

  retval *= 0.5F;
  return retval;
}

template<int BatchSize, int OutputSize>
eig::Array<float, BatchSize, OutputSize, eig::RowMajor>
loss_grad(const eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>& pred_out, 
          const eig::Array<float, BatchSize, OutputSize, eig::RowMajor>& ref_out)
{
  eig::Array<float, BatchSize, OutputSize, eig::RowMajor> retval;
  retval(all, 0) = (pred_out(all, 0) - ref_out(all, 0));
  retval(all, 1) = (pred_out(all, 1) - ref_out(all, 1));
  return retval;
}


int main()
{
  // auto world_barriers = ENV::read_world(WORLD_FILE);
  
  // TODO: add robot parameter file for experimentation 
  // auto robot_params   = ENV::read_robot_config("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/robot_params.yaml");

  ArtificialNeuralNetwork<2, 3, 3, 4, 5, 6, 2> ann;
  ann.dense(Activation(SIGMOID, HE), 
            Activation(SIGMOID, HE), 
            Activation(SIGMOID, HE), 
            Activation(SIGMOID, HE), 
            Activation(SIGMOID, HE), 
            Activation(SIGMOID, HE));

  eig::Array<float, 5000, 2, eig::RowMajor> X, Y;
  X.setRandom(); Y.setRandom();

  auto start = TIME_NOW;
  auto[weight_grad, bias_grad] = gradient_batch<5000>(ann, X, Y, loss, loss_grad<5000, 2>);
  auto end   = TIME_NOW;

  std::cout << "Elapsed: " << std::chrono::duration<float>(end - start).count();

  // std::cout << "\ndW: " << weight_grad << "\n";
  // std::cout << "\ndb: " << bias_grad << "\n";

  return EXIT_SUCCESS;
}