#include <iostream>
#include <random>

#include "global_typedef.h"
#include "environment_util.h"
#include "ANN/ANN.h"
#include "cppyplot.hpp"
#include "learning/to_drive.h"

using namespace ANN;
#define WORLD_FILE ("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/world_barriers.csv")
#define CONFIG_FILE ("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/global_params.yaml")

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
  auto global_config  = ENV::read_global_config(CONFIG_FILE);
  
  auto [actor_network, critic_network] = learn_to_drive(global_config);

  return EXIT_SUCCESS;
}