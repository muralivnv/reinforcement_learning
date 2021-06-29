#include <iostream>

#include "global_typedef.h"
#include "learning/to_drive/to_drive.h"

#include "util/environment_util.h"

// #include "ANN_v2/test_ANN.h"

using namespace ANN;
#define WORLD_FILE ("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/world_barriers.csv")
#define CONFIG_FILE ("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/global_params.yaml")

int main()
{
  // auto world_barriers = ENV::read_world(WORLD_FILE);
  auto global_config  = ENV::read_global_config(CONFIG_FILE);

  learn_to_drive(global_config);

  // test_ann<100>();

  return EXIT_SUCCESS;
}