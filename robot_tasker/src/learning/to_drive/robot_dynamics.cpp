#include "robot_dynamics.h"

namespace learning::to_drive
{

DifferentialRobotState differential_robot(const DifferentialRobotState& cur_state, 
                                          const WheelSpeeds&            cmd_vel, 
                                          const global_config_t&         global_config)
{
  DifferentialRobotState next_state;
  static const float& wheel_radius = global_config.at("robot/wheel_radius");
  static const float& base_length  = global_config.at("robot/base_length"); 
  static const float& dt           = global_config.at("cycle_time");

  float average_wheel_speed = 0.5F*(cmd_vel.left + cmd_vel.right);
  
  float vx      = wheel_radius*average_wheel_speed*std::cosf(cur_state.psi);
  float vy      = wheel_radius*average_wheel_speed*std::sinf(cur_state.psi);
  float psi_dot = (wheel_radius/base_length)*(cmd_vel.right - cmd_vel.left);

  next_state.x   = cur_state.x + (vx*dt);
  next_state.y   = cur_state.y + (vy*dt);
  next_state.psi = cur_state.psi + (psi_dot*dt);

  return next_state;
}

} // namespace {learning::to_drive}