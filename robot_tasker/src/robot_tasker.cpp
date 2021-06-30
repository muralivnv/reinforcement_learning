#include "global_typedef.h"
#include "learning/to_drive/to_drive.h"
#include "learning/to_drive/to_drive_visualization.h"

#include "util/environment_util.h"

using namespace ANN;
#define CONFIG_FILE ("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/global_params.yaml")

int main()
{
  auto global_config  = ENV::read_global_config(CONFIG_FILE);

  auto [actor_network, critic_network] = learn_to_drive(global_config);
  show_me_what_you_learned(actor_network, critic_network, CONFIG_FILE, 2u);

  return EXIT_SUCCESS;
}