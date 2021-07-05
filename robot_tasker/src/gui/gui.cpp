#include "gui.h"
#include "gui_typedef.h"
#include "gui_util.h"

#include <iostream>
#include <imgui.h>
#include <imgui-SFML.h>
#include <implot.h>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>

#include "../util/util.h"

namespace gui
{

WorldDrawParams world_draw_params;
GlobalContainer global_container(world_draw_params.window_width, 
                                 world_draw_params.window_height);

void initiate_gui(const string& global_config)
{
  auto config = util::read_global_config(global_config);
  
  // initiate window
  global_container.window.setFramerateLimit(world_draw_params.framerate);
  
  // initiate imgui
  ImGui::SFML::Init(global_container.window);
  ImPlot::CreateContext();

  // set world max bounds
  global_container.world_bounds_world_coord[X] = config.at("world/size/x");
  global_container.world_bounds_world_coord[Y] = config.at("world/size/y");
  
  // calculate world to sfml coord conversion factor
  global_container.world2sfml_factor[X] = (float)world_draw_params.window_width/global_container.world_bounds_world_coord[X];
  global_container.world2sfml_factor[Y] = (float)world_draw_params.window_height/global_container.world_bounds_world_coord[Y];
  calculate_sf_origin();

  // calculate max sf coordinates
  float max_x_sf_coord = (world_draw_params.scaling*world_draw_params.window_width) + global_container.world_origin_sf_coord[X];
  float max_y_sf_coord = (world_draw_params.scaling*world_draw_params.window_height) + world_draw_params.boundary_padding;

  // load font 
  auto this_file_folder = util::get_file_dir_path(__FILE__);
  if (!global_container.grid_tick_text_font.loadFromFile(this_file_folder + "/font.ttf"))
  {
    std::cerr << "Font load failed\n";
  }

  // calculate grid ticks and tick text
  tie(global_container.grid_ticks, 
      global_container.grid_tick_text) = generate_grid_data(max_x_sf_coord, max_y_sf_coord);
  
  // initiate robot
  // create marker for robot position
  auto [x_ini_pos, y_ini_pos] = world2sfml_position(50.0F, 50.0F);
  global_container.robot_position = sf::CircleShape{5, 3};
  global_container.robot_position.setFillColor(sf::Color::Magenta);
  global_container.robot_position.setPosition(x_ini_pos, y_ini_pos);

  // initiate target
  // create marker for target position
  tie (x_ini_pos, y_ini_pos) = world2sfml_position(124.0F, 678.0F);
  global_container.target_position = sf::CircleShape{5};
  global_container.target_position.setFillColor(sf::Color::Black);
  global_container.target_position.setPosition(x_ini_pos, y_ini_pos);
}



void gui_render_begin()
{
  auto& window = global_container.window;
  if (window.isOpen())
  {
    sf::Clock clock;
    sf::Event event;
    window.clear(world_draw_params.world_bg_color);
    while(window.pollEvent(event))
    {
      ImGui::SFML::ProcessEvent(event);
      if (event.type == sf::Event::Closed)
      {
        window.close();
      }
    }
    ImGui::SFML::Update(window, clock.restart());
    ImGui::Begin("World");
    static float current_scaling = world_draw_params.scaling;
    ImGui::SliderFloat("Scaling", &current_scaling, 0.25F, 0.85F);
    if ( fabsf(current_scaling - world_draw_params.scaling) > 0.05F)
    {
      scale_world(current_scaling);
    }
    
    ImGui::End();
  }
}

void gui_render_finalize()
{
  auto& window = global_container.window;
  if (window.isOpen())
  {
    // draw grid ticks
    std::for_each(global_container.grid_ticks.begin(), global_container.grid_ticks.end(), 
                  [&window](auto& vertices) {window.draw(vertices.data(), 2, sf::Lines);} );
    
    // draw grid tick texts
    std::for_each(global_container.grid_tick_text.begin(), global_container.grid_tick_text.end(), 
                  [&window](auto& text) {window.draw(text); });
    
    // draw robot position
    window.draw(global_container.robot_position);

    // draw target position
    window.draw(global_container.target_position);

    // finalize
    ImGui::SFML::Render(window);
    window.display();
  }
}

void gui_render_close()
{
  ImPlot::DestroyContext();
}

bool is_gui_opened()
{
  return global_container.window.isOpen();

}
} // namespace {gui}