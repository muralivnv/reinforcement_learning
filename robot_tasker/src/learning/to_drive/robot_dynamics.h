#ifndef _ROBOT_DYNAMICS_H_
#define _ROBOT_DYNAMICS_H_

#include "../../global_typedef.h"

namespace RL
{

DifferentialRobotState differential_robot(const RL::DifferentialRobotState& cur_state, 
                                              const RL::Action2D&               cmd_vel, 
                                              const RL::GlobalConfig_t&         global_config);

DifferentialRobotState 
robot_state_diff(const RL::DifferentialRobotState& lhs, 
                 const RL::DifferentialRobotState& rhs);

} // namespace {RL}

#endif
