#ifndef _ROBOT_DYNAMICS_H_
#define _ROBOT_DYNAMICS_H_

#include "../../global_typedef.h"
#include "robot_typedef.h"

namespace learning::to_drive
{
using namespace global_typedef;

DifferentialRobotState differential_robot(const DifferentialRobotState& cur_state, 
                                          const WheelSpeeds&            cmd_vel, 
                                          const global_config_t&         global_config);

} // namespace {learning::to_drive}

#endif
