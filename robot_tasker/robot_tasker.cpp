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

int main()
{
  // auto world_barriers = ENV::read_world(WORLD_FILE);
  auto global_config  = ENV::read_global_config(CONFIG_FILE);

  ENV::realtime_visualizer_init(WORLD_FILE);
  auto [actor_network, critic_network] = learn_to_drive(global_config);

  return EXIT_SUCCESS;
}