#ifndef _TO_DRIVE_TYPEDEF_H_
#define _TO_DRIVE_TYPEDEF_H_

namespace learning::to_drive
{

enum ReplayBufferIndices{
  S0 = 0,
  S1, 
  A0, 
  A1, 
  R, 
  NEXT_S0, 
  NEXT_S1,
  EPISODE_STATE,
  BUFFER_LEN,
};

struct TargetReachSuccessParams{
  float min_req_x_error_to_target;
  float min_req_y_error_to_target;
};

} // namespace {learning::to_drive}

#endif
