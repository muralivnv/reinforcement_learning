#ifndef _GUI_UTIL_H_
#define _GUI_UTIL_H_

#include "../global_typedef.h"
#include "gui_typedef.h"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Text.hpp>

namespace gui
{

std::string
to_string_with_precision(const float value, const int precision);

tuple<float, float>
world2sfml_position(const float x_world_coord, const float y_world_coord);

float world2sfml_heading(const float heading_rad_world);

tuple<std::vector<std::array<sf::Vertex, 2>>, std::vector<sf::Text>>
generate_grid_data(const float x_max_sf_coord, const float y_max_sf_coord);

void scale_world(const float scaling_factor);
void calculate_sf_origin();

void set_robot_state(const float x_world_coord, const float y_world_coord, const float psi_world_coord);
void set_target_state(const float x_world_coord, const float y_world_coord);

} // namespace {gui}

#endif
