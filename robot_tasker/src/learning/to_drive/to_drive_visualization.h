#ifndef _TO_DRIVE_VISUALIZATION_H_
#define _TO_DRIVE_VISUALIZATION_H_

#include <thread>

#include "../../global_typedef.h"
#include "../../ANN/ANN.h"

#include "robot_dynamics.h"
#include "to_drive_util.h"
#include "../../util/environment_util.h"

namespace learning::to_drive
{

template<typename ActorNetwork_t, typename CriticNetwork_t>
void show_me_what_you_learned(const ActorNetwork_t& actor, 
                              const CriticNetwork_t& critic, 
                              const std::string& config_file, 
                              const size_t           max_episodes)
{
  auto global_config  = env_util::read_global_config(config_file);
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");
  static const TargetReachSuccessParams target_reach_params_local = TargetReachSuccessParams{1.5F, deg2rad(30.0F)};

  Cppyplot::cppyplot pyp;

  std::random_device seed;
  std::mt19937 rand_gen(seed());
  std::uniform_real_distribution<float> state_x_sample(0.0F, world_max_x);
  std::uniform_real_distribution<float> state_y_sample(0.0F, world_max_y);
  std::uniform_real_distribution<float> state_psi_sample(-PI, PI);
  std::uniform_real_distribution<float> action1_sample(0.0F, action1_max);
  std::uniform_real_distribution<float> action2_sample(0.0F, action2_max);

  size_t episode_count = 0u;

  env_util::realtime_visualizer_init(config_file, 10u);
  while(episode_count <= max_episodes)
  {
    episode_count++;

    bool is_episode_done = false;
    DifferentialRobotState cur_state, target_state, next_state;
    eig::Array<float, 1, 2, eig::RowMajor> policy_s_now, policy_action, policy_s_next;
    eig::Array<float, 1, 4, eig::RowMajor> critic_input;

    tie(cur_state, target_state) = init_new_episode(state_x_sample, state_y_sample, state_psi_sample, rand_gen);
    env_util::update_target_pose({target_state.x, target_state.y});

    while (NOT(is_episode_done))
    {
      tie(policy_s_now(0,0), policy_s_now(0, 1)) = cur_state - target_state;
      state_normalize(global_config, policy_s_now);
      
      policy_action = forward_batch<1>(actor, policy_s_now);
      next_state  = differential_robot(cur_state, {policy_action(0, 0)*action1_max, policy_action(0, 1)*action2_max}, global_config);
      next_state.psi = util::wrapto_minuspi_pi(next_state.psi);
      
      critic_input(0, S0) = policy_s_now(0, 0);
      critic_input(0, S1) = policy_s_now(0, 1);
      critic_input(0, A0) = policy_action(0, 0);
      critic_input(0, A1) = policy_action(0, 1);
      auto Q   = forward_batch<1>(critic, critic_input);

      cur_state = next_state;

      is_episode_done = is_robot_outside_world(next_state, global_config);
      is_episode_done |= has_robot_reached_target(next_state, target_state, target_reach_params_local);

      auto [pose_error, heading_error] = next_state - target_state;
      std::cout << "episode: " << episode_count << ", pose_error: " << pose_error << ", heading_error: " << heading_error << '\n';

      env_util::update_visualizer({next_state.x, next_state.y}, 
                            {policy_action(0,0)*action1_max, policy_action(0,1)*action2_max}, 
                            Q(0, 0), 10);
      
      std::this_thread::sleep_for(10ms);
    }
  }
}

} // namespace {learning::to_drive}

#endif
