#include "global_typedef.h"
#include "learning/to_drive/to_drive.h"
#include "learning/to_drive/to_drive_visualization.h"

#include "util/util.h"
#include "gui/gui.h"

#include <imgui.h>
#include <imgui-SFML.h>
#include <implot.h>

using namespace ANN;
using namespace learning::to_drive;
using namespace gui;

#define CONFIG_FILE ("../global_params.yaml")

int main()
{
  auto config_full_path = util::get_file_dir_path(__FILE__) + "/" + CONFIG_FILE;
  initiate_gui(config_full_path);

  while(is_gui_opened())
  {
    gui_render_begin();

    ImGui::Begin("LearningToDrive");
    if (ImGui::Button("start_learning"))
    {
      ImGui::End();
      gui_render_finalize();


      auto global_config                   = util::read_global_config(config_full_path);
      auto [actor_network, critic_network] = learn_to_drive(global_config);
      show_me_what_you_learned(actor_network, critic_network, config_full_path, 3u);
    }
    else
    {
      ImGui::End();
    }

    gui_render_finalize();
  }
  gui_render_close();
  
  return EXIT_SUCCESS;
}