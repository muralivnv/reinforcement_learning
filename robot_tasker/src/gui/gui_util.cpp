#include "gui_util.h"
#include "gui_typedef.h"
#include <sstream>
#include <iomanip>

namespace gui
{

extern WorldDrawParams  world_draw_params;
extern GlobalContainer  global_container;

std::string 
to_string_with_precision(const float value, const int precision = 1)
{
  std::ostringstream out;
  out.precision(precision);
  out << std::fixed << value;
  return out.str();
}

tuple<float, float>
world2sfml_position(const float x_world_coord, const float y_world_coord)
{
  float x_sfml = world_draw_params.scaling*x_world_coord*global_container.world2sfml_factor[X]; 
  float y_sfml = -world_draw_params.scaling*y_world_coord*global_container.world2sfml_factor[Y];

  // transform with respect to required origin
  x_sfml += global_container.world_origin_sf_coord[X];
  y_sfml += global_container.world_origin_sf_coord[Y];

  return std::make_tuple(x_sfml, y_sfml);
}

float 
world2sfml_heading (const float heading_rad_world)
{
  float heading_sfml_deg = 0.0F;
  const float heading_deg_world = rad2deg(heading_rad_world);
  if (heading_deg_world > 90.0F)
  {
    heading_sfml_deg  = 450.0F - heading_deg_world;
  }
  else
  {
    heading_sfml_deg = 90.0F - heading_deg_world;
  }

  return heading_sfml_deg;
}


tuple<vector<array<sf::Vertex, 2>>, vector<sf::Text>>
generate_grid_data(const float x_max_sf_coord, 
                   const float y_max_sf_coord)
{
  tuple<vector<array<sf::Vertex, 2>>, vector<sf::Text>> out = make_tuple(vector<array<sf::Vertex, 2>>{}, 
                                                                         vector<sf::Text>{});
  auto& [grid_ticks, tick_text] = out;

  // generate x-grid lines
  const int n_ticks_x = (int)std::ceil(x_max_sf_coord/world_draw_params.grid_spacing);
  grid_ticks.reserve(n_ticks_x);
  for (float x_coord = global_container.world_origin_sf_coord[X]; x_coord < x_max_sf_coord; x_coord += world_draw_params.grid_spacing)
  {
    const float& x1 = x_coord;
    const float& y1 = global_container.world_origin_sf_coord[Y];
    const float& x2 = x_coord;
    const float y2  = y1 + world_draw_params.tick_len;

    grid_ticks.push_back(array<sf::Vertex, 2>{sf::Vertex(sf::Vector2f{x1, y1}), 
                                              sf::Vertex(sf::Vector2f{x2, y2})});
    grid_ticks.back()[0].color = world_draw_params.tick_color;
    grid_ticks.back()[1].color = world_draw_params.tick_color;
  }

  // generate y-grid lines
  const int n_ticks_y = (int)std::ceil(y_max_sf_coord/world_draw_params.grid_spacing);
  grid_ticks.reserve(n_ticks_y);
  for (float y_coord = world_draw_params.boundary_padding; y_coord < y_max_sf_coord; y_coord += world_draw_params.grid_spacing)
  {
    const float& x1 = global_container.world_origin_sf_coord[X];
    const float& y1 = y_coord;
    const float  x2 = x1 - world_draw_params.tick_len;
    const float& y2 = y_coord;  

    grid_ticks.push_back(array<sf::Vertex, 2>{sf::Vertex(sf::Vector2f{x1, y1}), 
                                              sf::Vertex(sf::Vector2f{x2, y2})});
    grid_ticks.back()[0].color = world_draw_params.tick_color;
    grid_ticks.back()[1].color = world_draw_params.tick_color;
  }

  // draw world bounds
  grid_ticks.push_back(array<sf::Vertex, 2>{sf::Vertex(sf::Vector2f{global_container.world_origin_sf_coord[X], 
                                                                    global_container.world_origin_sf_coord[Y]}), 
                                            sf::Vertex(sf::Vector2f{global_container.world_origin_sf_coord[X] + x_max_sf_coord, 
                                                                    global_container.world_origin_sf_coord[Y]})});
  grid_ticks.back()[0].color = world_draw_params.tick_color;
  grid_ticks.back()[1].color = world_draw_params.tick_color;
  
  // axis max tick text
  tick_text.push_back(sf::Text());
  sf::Text* this_text = &tick_text.back();
  this_text->setPosition(global_container.world_origin_sf_coord[X] + x_max_sf_coord, 
                         global_container.world_origin_sf_coord[Y] + world_draw_params.tick_len + 10.0F);
  this_text->setString("X-" + to_string_with_precision(global_container.world_bounds_world_coord[X], 0) + " (m)");
  this_text->setFillColor(sf::Color::Black);
  this_text->setStyle(sf::Text::Bold);
  this_text->setCharacterSize(12);
  this_text->setFont(global_container.grid_tick_text_font);
  
  grid_ticks.push_back(array<sf::Vertex, 2>{sf::Vertex(sf::Vector2f{global_container.world_origin_sf_coord[X], 
                                                                    global_container.world_origin_sf_coord[Y]}), 
                                            sf::Vertex(sf::Vector2f{global_container.world_origin_sf_coord[X], 
                                                                    world_draw_params.boundary_padding})});                                                               
  grid_ticks.back()[0].color = world_draw_params.tick_color;
  grid_ticks.back()[1].color = world_draw_params.tick_color;
  
  // axis max tick text
  tick_text.push_back(sf::Text());
  this_text = &tick_text.back();
  this_text->setPosition(global_container.world_origin_sf_coord[X] - world_draw_params.tick_len - 10.0F, 
                         world_draw_params.boundary_padding-10.0F);
  this_text->setString("Y-" + to_string_with_precision(global_container.world_bounds_world_coord[Y], 0) + " (m)");
  this_text->setFillColor(sf::Color::Black);
  this_text->setStyle(sf::Text::Bold);
  this_text->setCharacterSize(12);
  this_text->setFont(global_container.grid_tick_text_font);
  grid_ticks.shrink_to_fit();

  return out;
}


void scale_world(const float scaling_factor)
{
  sf::Vector2u window_size = global_container.window.getSize();
  world_draw_params.scaling = scaling_factor;
  calculate_sf_origin();

  float max_x_sf_coord = (world_draw_params.scaling*world_draw_params.window_width) + global_container.world_origin_sf_coord[X];
  float max_y_sf_coord = (world_draw_params.scaling*world_draw_params.window_height) + world_draw_params.boundary_padding;

  tie(global_container.grid_ticks, global_container.grid_tick_text) = generate_grid_data(max_x_sf_coord, max_y_sf_coord);
}


void calculate_sf_origin()
{
  global_container.world_origin_sf_coord[X] = world_draw_params.boundary_padding;
  global_container.world_origin_sf_coord[Y] = (world_draw_params.scaling*world_draw_params.window_height) 
                                              +  world_draw_params.boundary_padding;
}


void set_robot_state(const float x_world_coord, const float y_world_coord, const float psi_world_coord)
{
  auto [x_robot, y_robot] = world2sfml_position(x_world_coord, y_world_coord);
  auto heading_deg        = world2sfml_heading(psi_world_coord);

  global_container.robot_position.setPosition(x_robot, y_robot);
  global_container.robot_position.setRotation(heading_deg);
}


void set_target_state(const float x_world_coord, const float y_world_coord)
{
  auto [x_target, y_target] = world2sfml_position(x_world_coord, y_world_coord);
  global_container.target_position.setPosition(x_target, y_target);
}


} // namespace {gui}