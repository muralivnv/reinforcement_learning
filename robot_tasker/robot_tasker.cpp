#include <iostream>
#include <random>

#include "global_typedef.h"
#include "environment_util.h"
#include "ANN/ANN.h"
#include "cppyplot.hpp"

using namespace ANN;
#define WORLD_FILE ("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/world_barriers.csv")
int main()
{
  // auto world_barriers = ENV::read_world(WORLD_FILE);
  
  // TODO: add robot parameter file for experimentation 
  // auto robot_params   = ENV::read_robot_config("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/robot_params.yaml");

  // ArtificialNeuralNetwork<2, 3, 3, 2> sampling_policy;
  // sampling_policy.dense(ReLU(Initializer(HE)), 
  //                       ReLU(Initializer(HE)), 
  //                       ReLU(Initializer(HE)) );
  
  // RL::Matrix<float, 25, 2> X;
  // RL::Matrix<float, 25, 2> Y;
  // auto out = gradient_batch<25>(sampling_policy, X, Y);

  // TODO: Make it compile
  // TODO: Test ANN with a sample test
  ENV::realtime_visualizer_init(WORLD_FILE);
  float x = 0.0F, y = 0.0F, Vx = 4.5F, Vy = 2.5F;
  std::random_device seed;
  std::mt19937 rand_gen(seed());
  std::uniform_real_distribution<float> uniform_dist(0.0F, 4.5F);

  for (float t = 0.04F; t < 10.0F; t+=0.04F)
  {
    Vx = uniform_dist(rand_gen);
    Vy = uniform_dist(rand_gen);
    x += Vx*t;
    y += Vy*t;
    ENV::update_visualizer({x,y}, {Vx, Vy}, uniform_dist(rand_gen));
    std::this_thread::sleep_for(10ms);
  }

  return EXIT_SUCCESS;
}