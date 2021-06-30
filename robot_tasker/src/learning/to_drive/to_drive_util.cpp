#include "to_drive_util.h"
#include "../../util/util.h"

namespace learning::to_drive
{

using namespace global_typedef;

tuple<float, float> operator-(const DifferentialRobotState& actual, 
                              const DifferentialRobotState& reference)
{
  const float range_error = std::sqrtf( util::squaref(actual.x - reference.x) 
                                      + util::squaref(actual.y - reference.y) );

  const float heading_req = std::atan2f( (reference.y - actual.y), 
                                         (reference.x - actual.x));

  const float heading_error = util::wrapto_minuspi_pi(actual.psi - heading_req);
  
  return std::make_tuple(range_error, heading_error);
}

tuple<DifferentialRobotState, DifferentialRobotState>
init_new_episode(std::uniform_real_distribution<float>& state_x_sample, 
                 std::uniform_real_distribution<float>& state_y_sample, 
                 std::uniform_real_distribution<float>& state_psi_sample, 
                 std::mt19937& rand_gen)
{
  tuple ret_val = std::make_tuple(DifferentialRobotState(), DifferentialRobotState());
  auto& [init_state, final_state] = ret_val;
  
  init_state.x   = state_x_sample(rand_gen);
  init_state.y   = state_y_sample(rand_gen);
  init_state.psi = state_psi_sample(rand_gen);

  final_state.x   = state_x_sample(rand_gen);
  final_state.y   = state_y_sample(rand_gen);
  final_state.psi = state_psi_sample(rand_gen);

  return ret_val;
}


float get_exploration_noise(std::normal_distribution<float>& exploration_noise_dist, 
                           std::mt19937& rand_gen)
{
  const float exploration_noise = exploration_noise_dist(rand_gen);
  return exploration_noise;
}

void state_normalize(const GlobalConfig_t&               global_config, 
                     eig::Array<float, 1, 2, eig::RowMajor>& policy_state)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float max_range_error = std::sqrtf(  (world_max_x)*(world_max_x) 
                                                  + (world_max_y)*(world_max_y) );
  static const float max_heading_error = PI;

  policy_state(0, 0) /= max_range_error;
  policy_state(0, 1) /= max_heading_error;
}

bool is_robot_outside_world(const DifferentialRobotState& state,
                            const GlobalConfig_t&         global_config)
{
  static const float& world_max_x = global_config.at("world/size/x");
  static const float& world_max_y = global_config.at("world/size/y"); 

  if (   (state.x > 0.0F) && (state.x < world_max_x)
      && (state.y > 0.0F) && (state.y < world_max_y) )
  {
    return false;
  }
  return true;
}

bool has_robot_reached_target(const DifferentialRobotState& current_state, 
                              const DifferentialRobotState& target_state, 
                              const TargetReachSuccessParams&   target_reached_criteria)
{
  bool is_reached = false;
  auto [range_error, heading_error] = current_state - target_state;

  if (   (range_error          < target_reached_criteria.min_req_range_error_to_target  )
      && (fabsf(heading_error) < target_reached_criteria.min_req_heading_error_to_target) )
  {
    is_reached = true;
  }
  return is_reached;
}

} // namespace {learning::to_drive}
