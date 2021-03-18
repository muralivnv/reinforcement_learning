#ifndef _TO_DRIVE_H_
#define _TO_DRIVE_H_

#include <random>
#include <algorithm>

#include "../global_typedef.h"
#include "../ANN/ANN_typedef.h"
#include "../ANN/ANN.h"
#include "../ANN/ANN_optimizers.h"

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
  auto state_error    = current_state - target_state;
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

auto reset_states(std::uniform_real_distribution& pose_x_sample, 
                  std::uniform_real_distribution& pose_y_sample, 
                  std::uniform_real_distribution& heading_sample, 
                  std::mt19937&                   gen)
{
  RL::DifferentialRobotState initial_state, final_state;
  initial_state.x = pose_x_sample(gen);
  initial_state.y = pose_y_sample(gen);
  initial_state.phi = heading_sample(gen);

  final_state.x = pose_x_sample(gen);
  final_state.y = pose_y_sample(gen);
  final_state.phi = heading_sample(gen);

  return std::make_tuple(initial_state, final_state);
}

template<typename EigenDerived1, typename EigenDerived2>
float critic_loss(const eig::ArrayBase<EigenDerived1>& Q_sampling, 
                  const eig::ArrayBase<EigenDerived2>& Q_target)
{
  float loss = -0.5F*((Q_sampling - Q_target).square()).mean();
  return loss;
}

template<int BatchSize>
eig::Array<float, BatchSize, 1, eig::RowMajor>
critic_loss_grad(const eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>& Q_sampling, 
                 const eig::Array<float, BatchSize, 1, eig::RowMajor>& Q_target)
{
  // Loss = -(0.5/BatchSize)*Sum( Square(Q_sampling - Q_target) )

  eig::Array<float, BatchSize, OutputSize, eig::RowMajor> retval = (Q_target - Q_sampling);
  return retval;
}

bool is_robot_inside_world(const RL::DifferentialRobotState& state,
                           const RL::GlobalConfig_t&         global_config)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 

  if (   (state.x > 0.0F) && (state.x < world_max_x)
      && (state.y > 0.0F) && (state.y < world_max_y) )
  {
    return true;
  }
  return false;
}

auto learn_to_drive(const RL::GlobalConfig_t& global_config)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");
  const float& dt = global_config.at("cycle_time");
  const int s0 = 0;
  const int s1 = 1;
  const int a0 = 2;
  const int a1 = 3;
  const int r  = 4;
  const int next_s0 = 5;
  const int next_s1 = 6;

  std::random_device seed;
  std::mt19937 rand_gen(seed());
  std::uniform_real_distribution pose_x_sample(0, world_max_x);
  std::uniform_real_distribution pose_y_sample(0, world_max_y);
  std::uniform_real_distribution heading_sample(-PI, PI);

  constexpr size_t batch_size   = 256u;
  constexpr size_t max_episodes = 100u;
  constexpr size_t replay_buffer_size = 1000u;

  eig::<float, eig::Dynamic, 7, eig::RowMajor, replay_buffer_size> replay_buffer;
  int replay_buffer_len = -1;
  size_t episode_count  = 0u; 
  std::normal_distribution<float> exploration_noise(0.0F, 0.1F);

  // Deep deterministic policy gradient
  ArtificialNeuralNetwork<2, 3, 5, 2> sampling_actor, target_actor;
  ArtificialNeuralNetwork<4, 5, 7, 1> sampling_critic, target_critic;

  sampling_actor.dense(Activation(RELU, HE), 
                        Activation(RELU, HE), 
                        Activation(SIGMOID, XAVIER));
  target_actor = sampling_actor;
  
  sampling_critic.dense(Activation(RELU, HE), 
                        Activation(RELU, HE), 
                        Activation(SIGMOID, XAVIER));
  target_critic = sampling_critic;

  OptimizerParams actor_opt;
  actor_opt["step_size"] = 1e-3F;

  OptimizerParams critic_opt;
  critic_opt["step_size"] = 1e-4f;
  
  float soft_update_rate = 0.95F;
  while (episode_count < max_episodes)
  {
    auto [current_state, target_state] = reset_states(pose_x_sample, pose_y_sample, heading_sample, rand_gen);

    eig::Array<float, 1, 2, eig::RowMajor> state;
    eig::Array<float, 1, 2, eig::RowMajor> action;
    float reward;
    eig::Array<float, 1, 2, eig::RowMajor> next_state;

    bool robot_state_inside_world = true;
    size_t current_cycle          = 0u;
    while(robot_state_inside_world)
    {
      // select action based on the currrent state of the robot
      [state(0, 0), state(0, 1)] = calc_error(current_state, target_state);
      action  = forward_batch<1>(sampling_actor, state);
      action += exploration_noise(rand_gen);

      // execute the action and observe next state, reward
      auto state_projected = differential_robot(current_state, 
                                               {action(0, 0), action(0, 1)}, dt);

      [next_state(0, 0), next_state(0, 1)] = calc_error(state_projected, target_state);
      reward = calc_reward(state_projected - target_state);

      // store current transition -> S_t, A_t, R_t, S_{t+1} in replay buffer
      replay_buffer_len++;
      replay_buffer_len %= replay_buffer_size;
      if (replay_buffer.rows() < replay_buffer_len)
      { replay_buffer.conservativeResize(replay_buffer_len+1, NoChange); }
      replay_buffer(replay_buffer_len, {s0, s1})           = state;
      replay_buffer(replay_buffer_len, {a0, a1})           = action;
      replay_buffer(replay_buffer_len, r)                  = reward;
      replay_buffer(replay_buffer_len, {next_s0, next_s1}) = next_state;

      state = next_state;
      current_cycle++;

      if (current_cycle >= batch_size)
      {
        // Sample 'N' transitions from replay buffer to do mini-batch param optimization
        auto n_transitions = get_n_shuffled_idx<batch_size>(replay_buffer.rows());

        eig::Array<float, batch_size, 1> Q_target;
        eig::Array<float, batch_size, 4, eig::RowMajor> target_critic_input;

        target_critic_input(all, {s0, s1}) = replay_buffer(n_transitions, {next_s0, next_s1});
        target_critic_input(all, {a0, a1}) = forward_batch<batch_size>(target_actor, target_critic_input(all, {s0, s1}) );

        Q_target  = forward_batch<batch_size>(target_critic, target_critic_input);
        Q_target *= discount_factor; // TODO: define this
        Q_target += replay_buffer(n_transitions, r);

        // calculate loss between Q_target, Q_sampling and perform optimization step
        auto [loss, critic_weight_grad, critic_bias_grad] = gradient_batch<batch_size>(sampling_critic, 
                                                                                       replay_buffer(n_transitions, {s0, s1, a0, a1}, 
                                                                                       Q_target, 
                                                                                       critic_loss,
                                                                                       critic_loss_grad<batch_size>);
        steepest_descent(critic_weight_grad, critic_bias_grad, critic_opt, sampling_critic);

        // calculate loss for sampling_actor network and perform optimization step

        // soft-update target networks parameters
        target_actor.weight *= soft_update_rate;
        target_actor.weight += (1.0F - soft_update_rate)*sampling_actor.weight;

        target_critic.bias *= soft_update_rate;
        target_critic.bias += (1.0F - soft_update_rate)*sampling_critic.bias;
      }

      robot_state_inside_world = is_robot_inside_world(state, global_config);
    }
  }

  std::make_tuple(target_actor, target_critic);
}

#endif
