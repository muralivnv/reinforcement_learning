#ifndef _TO_DRIVE_UTIL_H_
#define _TO_DRIVE_UTIL_H_

#include <Eigen/Core>
#include <limits>
#include <random>
#include <cmath>

#include "../global_typedef.h"

#include "../ANN/ANN_activation.h"
#include "../ANN/ANN.h"
#include "../ANN/ANN_optimizers.h"

#include "../util.h"
#include "robot_dynamics.h"

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
  float min_req_range_error_to_target;
  float min_req_heading_error_to_target;
};

tuple<float, float> operator-(const RL::DifferentialRobotState& lhs, 
                              const RL::DifferentialRobotState& rhs);

tuple<RL::DifferentialRobotState, RL::DifferentialRobotState>
init_new_episode(std::uniform_real_distribution<float>& state_x_sample, 
                 std::uniform_real_distribution<float>& state_y_sample, 
                 std::uniform_real_distribution<float>& state_psi_sample, 
                 std::mt19937& rand_gen);

void add_exploration_noise(std::normal_distribution<float>& exploration_noise_dist, 
                           std::mt19937& rand_gen, 
                           eig::Array<float, 1, 2, eig::RowMajor>& action);

void state_normalize(const RL::GlobalConfig_t&               global_config, 
                     eig::Array<float, 1, 2, eig::RowMajor>& policy_state);

bool is_robot_outside_world(const RL::DifferentialRobotState& state,
                            const RL::GlobalConfig_t&         global_config);

bool has_robot_reached_target(const RL::DifferentialRobotState& current_state, 
                              const RL::DifferentialRobotState& target_state, 
                              const TargetReachSuccessParams&   target_reached_criteria);

template<int InputSize, int ... NHiddenLayers>
soft_update_network(const ANN::ArtificialNeuralNetwork<InputSize, NHiddenLayers...>&  target, 
                    const float smoothing_constant, 
                    ANN::ArtificialNeuralNetwork<InputSize, NHiddenLayers...>&  smoothened_network)
{
  // soft update weights
  const float target_smoothing_factor = 1.0F - smoothing_constant;
  smoothened_network.weight  = (smoothing_constant*smoothened_network.weight) + (target_smoothing_factor*target.weight);

  // soft update bias
  smoothened_network.bias  = (smoothing_constant*smoothened_network.bias) + (target_smoothing_factor*target.bias);
}

#endif
