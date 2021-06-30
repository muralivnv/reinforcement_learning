#include "global_typedef.h"
#include "learning/to_drive/to_drive.h"
#include "learning/to_drive/to_drive_visualization.h"

#include "util/util.h"
#include "util/environment_util.h"

using namespace ANN;
using namespace learning::to_drive;

#define CONFIG_FILE ("../global_params.yaml")

int main()
{
  auto config_full_path = util::get_file_dir_path(__FILE__) + "/" + CONFIG_FILE;
  auto global_config = env_util::read_global_config(config_full_path);

  auto [actor_network, critic_network] = learn_to_drive(global_config);
  show_me_what_you_learned(actor_network, critic_network, config_full_path, 2u);

  return EXIT_SUCCESS;
}