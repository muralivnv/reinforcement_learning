#ifndef _GUI_TYPEDEF_H_
#define _GUI_TYPEDEF_H_

#include <list>

#include "../global_typedef.h"
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Text.hpp>

namespace gui
{

enum WorldIndexing
{
  X = 0,
  Y
};

struct WorldDrawParams
{
  float grid_spacing     = 25.0F;   // sfml units
  float tick_len         = 4.0F;    // sfml units
  float boundary_padding = 35.0F;   // sfml units
  float scaling          = 0.35F;   // sfml units
  sf::Color tick_color     = sf::Color::Color(0, 0, 0, 170);
  sf::Color world_bg_color = sf::Color::Color(247, 247, 247);
  
  int window_width = 980;
  int window_height = 680;
  int framerate = 1000;
};

struct GlobalContainer
{
  sf::RenderWindow      window;
  std::array<float, 2>  world2sfml_factor; 
  std::array<float, 2>  world_origin_sf_coord;
  std::array<float, 2>  world_bounds_world_coord;
  std::vector<std::array<sf::Vertex, 2>> grid_ticks;
  std::vector<sf::Text> grid_tick_text;
  sf::Font grid_tick_text_font;

  sf::CircleShape target_position;
  sf::CircleShape robot_position;

  GlobalContainer(const int width, const int height, const std::string& title = "Robot Tasker"): window(sf::RenderWindow(sf::VideoMode(width, height), 
                                                                                                        title,  
                                                                                                        sf::Style::Default, 
                                                                                                        sf::ContextSettings(0, 0, 2))){}
};

} // namespace {gui}
#endif
