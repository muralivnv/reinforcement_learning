#ifndef _TO_DRIVE_H_
#define _TO_DRIVE_H_

#include <random>
#include <algorithm>

#include "../global_typedef.h"
#include "../ANN/ANN.h"
#include "../util.h"

#include "robot_dynamics.h"

using namespace ANN;

float calc_reward(const RL::DifferentialState& state_error)
{
  float pose_error    = std::sqrtf( (state_error.x*state_error.x) + (state_error.y*state_error.y) );
  float heading_error = state_error.psi;

  // calculate reward for position error
  float reward = linear_interpolate(fabsf(pose_error), 0.1F,  -0.05F, 1.0F,  -5.0F);

  // calculate reward for heading error
  reward += linear_interpolate(fabsf(heading_error),   0.01F, -0.05F, 0.05F, -5.0F);

  return reward;
}


auto calc_error(const RL::DifferentialState& current_state, 
                const RL::DifferentialState& target_state)
{
  auto state_error = current_state - target_state;
  float pose_error    = std::sqrtf( (state_error.x*state_error.x) + (state_error.y*state_error.y) );
  float heading_error = state_error.psi;

  return std::make_tuple(pose_error, heading_error);
}

template<int N>
std::array<int, N> get_n_shuffled_idx(const int container_size)
{
  std::vector<int> indices(container_size);
  std::array<int, N> shuffled_n_idx;
  std::random_device rd;
  std::mt19937 g(rd());

  std::generate(indices.begin(), indices.end(), [n = 0] () mutable { return n++; });
  std::shuffle(indices.begin(), indices.end(), g);

  for (int i = 0; i < N; i++)
  { shuffled_n_idx[i] = indices[i]; }

  return shuffled_n_idx;
}

// TODO: Output of this function should be a policy
auto learn_to_drive(const RL::GlobalConfig_t& global_config)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");

  std::random_device seed;
  std::mt19937 rand_gen(seed());
  std::uniform_real_distribution world_x(0, world_max_x);
  std::uniform_real_distribution world_y(0, world_max_y);
  std::uniform_real_distribution heading(-PI, PI);
  
  // Deep deterministic policy gradient
  ArtificialNeuralNetwork<2, 3, 5, 2> sampling_policy, target_policy;
  ArtificialNeuralNetwork<4, 5, 7, 1> sampling_action_value, target_action_value;

  sampling_policy.dense(Activation(RELU, HE), 
                        Activation(RELU, HE), 
                        Activation(SIGMOID, XAVIER));
  target_policy.dense(Activation(RELU, HE), 
                      Activation(RELU, HE), 
                      Activation(SIGMOID, XAVIER));
  
  sampling_action_value.dense(Activation(RELU, HE), 
                              Activation(RELU, HE), 
                              Activation(SIGMOID, XAVIER));
  target_action_value.dense(Activation(RELU, HE), 
                            Activation(RELU, HE), 
                            Activation(SIGMOID, XAVIER));

  eig::<float, 50, 7, eig::RowMajor> replay_buffer;
  int replay_buffer_len = -1;
  size_t max_episodes = 50u, episode_count = 0u;
  std::normal_distribution<float> exploration_noise(0.0F, 0.1F);

  while (episode_count < max_episodes)
  {
    RL::DifferentialRobotState current_state, target_state;
    current_state.x   = world_x(rand_gen);
    current_state.y   = world_y(rand_gen);
    current_state.psi = heading(rand_gen);
    target_state.x    = world_x(rand_gen);
    target_state.y    = world_y(rand_gen);
    target_state.psi  = heading(rand_gen);

    eig::Array<float, 1, 2, eig::RowMajor> policy_state;
    eig::Array<float, 1, 2, eig::RowMajor> policy_action;
    eig::Array<float, 1, 2, eig::RowMajor> transitioned_state;
    int warm_up_cycles = 25;
    float transition_reward;

    bool reach_to_target = true;
    size_t current_cycle = 0u;
    while(reach_to_target)
    {
      // select action based on the currrent state of the robot
      [policy_state(0, 0), policy_state(0, 1)] = calc_error(current_state, target_state);
      policy_action  = forward_batch<1>(sampling_policy, policy_state);
      policy_action += exploration_noise(rand_gen);

      // execute the action and observe next state, reward
      auto next_state = differential_robot(current_state, 
                                           {policy_action(0, 0), policy_action(0, 1)}, 0.04F);

      [transitioned_state(0, 0), transitioned_state(0, 1)] = calc_error(next_state, target_state);
      transition_reward = calc_reward(next_state - target_state);

      // store current transition -> S_t, A_t, R_t, S_{t+1} in replay buffer
      replay_buffer_len++;
      replay_buffer_len %= (replay_buffer.rows());
      replay_buffer(replay_buffer_len, 0) = policy_state(0, 0);
      replay_buffer(replay_buffer_len, 1) = policy_state(0, 1);
      replay_buffer(replay_buffer_len, 2) = policy_action(0, 0);
      replay_buffer(replay_buffer_len, 3) = policy_action(0, 1);
      replay_buffer(replay_buffer_len, 4) = transition_reward;
      replay_buffer(replay_buffer_len, 5) = transitioned_state(0, 0);
      replay_buffer(replay_buffer_len, 6) = transitioned_state(0, 1);

      if (current_cycle > warm_up_cycles)
      {
        // Sample 'N' transitions from replay buffer to do mini-batch param optimization
        auto n_transitions = get_n_shuffled_idx<25>(replay_buffer.rows()); // TODO: fix size issue

        eig::Array<float, 25, 1> G_target_policy;
        eig::Array<float, 25, 1> G_sampling_policy;
        eig::Array<float, 25, 4> action_value_input;

        auto target_action = forward_batch<25>(target_policy, replay_buffer(n_transitions, seq(5, 6)));
        action_value_input(all, seq(0, 1)) = replay_buffer(n_transitions, seq(5, 6));
        action_value_input(all, seq(2, 3)) = target_action;
        G_target_policy = forward_batch<25>(target_action_value, action_value_input);
        G_target_policy = replay_buffer(n_transitions, 4) + discount_factor*(G_target_policy);

        G_sampling_policy = forward_batch<25>(sampling_action_value, replay_buffer(n_transitions, seq(0, 3)));

        // calculate loss between G_target_policy and G_sampling_policy

        // calculate loss for sampling_policy network

        // update sampling_policy parameters

        // update sampling_action_value parameters

        // update target networks parameters
      }
      current_cycle++;
    }
  }
}

#endif
