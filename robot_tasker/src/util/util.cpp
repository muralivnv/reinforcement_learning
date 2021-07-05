#include "util.h"
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace util
{

namespace filesystem = std::filesystem;

float linear_interpolate(const float x, 
                         const float x1, const float y1, 
                         const float x2, const float y2)
{
  float interpolated_val;

  if (x < x1)
  {  interpolated_val = y1;  }
  else if (x > x2)
  {  interpolated_val = y2;  }
  else
  {
    interpolated_val = y1 + (x - x1)*(y2 - y1)/(x2 - x1);
  }
  return interpolated_val;
}

float wrapto_minuspi_pi(float angle) noexcept
{
  float normalized_angle = fmodf(angle, TWO_PI);
  if (PI < normalized_angle)
  {
    normalized_angle -= TWO_PI;
  }
  else if (normalized_angle < -PI)
  {
    normalized_angle += TWO_PI;
  }
  return normalized_angle;
}

float squaref(float x) noexcept
{
  return x*x;
}

std::string get_file_dir_path(const std::string& filename)
{
  std::filesystem::path file(filename);
  return file.parent_path().string();
}

std::unordered_map<std::string, float>
read_global_config(const std::string& config_name)
{
  std::unordered_map<std::string, float> retval;
  yml::Node config = yml::LoadFile(config_name);

  yml::Node node = config["world"]["size"];
  retval.insert(std::make_pair("world/size/x", node[0].as<float>()));
  retval.insert(std::make_pair("world/size/y", node[1].as<float>()));

  yml::Node robot_config = config["robot"];
  retval.insert(std::make_pair("robot/max_wheel_speed", robot_config["max_wheel_speed"].as<float>()));
  retval.insert(std::make_pair("robot/wheel_radius", robot_config["wheel_radius"].as<float>()));
  retval.insert(std::make_pair("robot/base_length", robot_config["base_length"].as<float>()));

  retval.insert(std::make_pair("cycle_time", config["cycle_time"].as<float>()));
  
  return retval;
}

} //namespace {util}