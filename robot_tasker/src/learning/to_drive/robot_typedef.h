#ifndef _ROBOT_TYPEDEF_H_
#define _ROBOT_TYPEDEF_H_

namespace learning::to_drive 
{

struct DifferentialRobotState{
  float x;   // (m)
  float y;   // (m)
  float psi; // heading (rad)
};

struct WheelSpeeds{
  float left;  // (m/sec)
  float right; // (m/sec)
};


} // namespace {learning::to_drive}

#endif
