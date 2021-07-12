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

// custom SFML shapes
class RobotShape: public sf::Shape
{
  private:
    array<float, 2> position_;
    float heading_rad_;
  
  protected:
    const float marker_radius_    = 5.0F;
    const float heading_dir_len_  = 15.0F;

  public:
    explicit RobotShape() {}
    
    void setState(const float x, const float y, const float heading_rad)
    {
      position_[X] = x;
      position_[Y] = y;
      heading_rad_ = heading_rad;

      this->update();
    }

    virtual size_t getPointCount() const
    {  return 6u;  }

    virtual sf::Vector2f getPoint(size_t index) const
    {
      sf::Vector2f this_point;
      
      // first 5 points will draw rhombus marker of robot position
      if (index < 5u)
      {
        float angle_rad = heading_rad_ + ((float)index * PI/2.0F);
        this_point.x = position_[X] + (std::cosf(angle_rad) * marker_radius_);
        this_point.y = position_[Y] - (std::sinf(angle_rad) * marker_radius_);
      }
      // last point will draw line representing robot heading
      else
      {
        this_point.x = position_[X] + (std::cosf(heading_rad_)*heading_dir_len_);
        this_point.y = position_[Y] - (std::sinf(heading_rad_)*heading_dir_len_);
      }
      return this_point;
    }
};

// custom SFML shapes
class TargetShape: public sf::Shape
{
  private:
    array<float, 2> position_;
  
  protected:
    const float line_len_ = 10.0F;

  public:
    explicit TargetShape() {}
    
    void setState(const float x, const float y)
    {
      position_[X] = x;
      position_[Y] = y;

      this->update();
    }

    virtual size_t getPointCount() const
    {  return 6u;  }

    virtual sf::Vector2f getPoint(size_t index) const
    {
      sf::Vector2f this_point(position_[X], position_[Y]); // T intersection point

      switch(index)
      {
        case 0u:
        case 5u:
        {
          this_point.y += line_len_; // T bottom half point
          break;
        }
        case 2u:
        {
          this_point.x += (0.5F*line_len_); // T left half point
          break;
        }
        case 3u:
        {
          this_point.x -= (0.5F*line_len_); // T right half point
          break;
        }
        default:
        { break; }
      }

      return this_point;
    }
};


struct WorldDrawParams
{
  float grid_spacing     = 25.0F;   // sfml units
  float tick_len         = 4.0F;    // sfml units
  float boundary_padding = 35.0F;   // sfml units
  float scaling          = 0.35F;   // sfml units
  sf::Color tick_color     = sf::Color::Color(0, 0, 0, 170);
  sf::Color world_bg_color = sf::Color::Color(247, 247, 247);
  sf::Color robot_color    = sf::Color::Color(38, 128, 196);
  
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

  RobotShape robot_position;
  TargetShape target_position;

  GlobalContainer(const int width, const int height, const std::string& title = "Robot Tasker"): window(sf::RenderWindow(sf::VideoMode(width, height), 
                                                                                                        title,  
                                                                                                        sf::Style::Default, 
                                                                                                        sf::ContextSettings(0, 0, 2))){}
};

} // namespace {gui}
#endif
