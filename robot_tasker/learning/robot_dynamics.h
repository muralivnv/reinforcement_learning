#ifndef _ROBOT_DYNAMICS_H_
#define _ROBOT_DYNAMICS_H_

#include "../global_typedef.h"

RL::DifferentialRobotState differential_robot(const RL::DifferentialRobotState& cur_state, 
                                              const RL::Action2D&               cmd_vel, 
                                              const RL::GlobalConfig_t&         global_config)
{
  RL::DifferentialRobotState next_state;
  static const float& wheel_radius = global_config.at("robot/wheel_radius");
  static const float& base_length  = global_config.at("robot/base_length"); 
  static const float& dt           = global_config.at("cycle_time");

  float average_wheel_speed = 0.5F*(cmd_vel.action1 + cmd_vel.action2);
  
  float vx      = wheel_radius*average_wheel_speed*std::cosf(cur_state.psi);
  float vy      = wheel_radius*average_wheel_speed*std::sinf(cur_state.psi);
  float psi_dot = (wheel_radius/base_length)*(cmd_vel.action2 - cmd_vel.action1);

  next_state.x   = cur_state.x + (vx*dt);
  next_state.y   = cur_state.y + (vy*dt);
  next_state.psi = cur_state.psi + (psi_dot*dt);

  return next_state;
}

RL::DifferentialRobotState operator-(const RL::DifferentialRobotState& lhs, const RL::DifferentialRobotState& rhs)
{
  RL::DifferentialRobotState diff;
  diff.x   = lhs.x - rhs.x;
  diff.y   = lhs.y - rhs.y;
  diff.psi = lhs.psi - rhs.psi;

  return diff;
}


#endif
