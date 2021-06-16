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
#include "to_drive_util.h"

using namespace ANN;

// local parameters
static const TargetReachSuccessParams target_reach_params = {.min_req_range_error_to_target = 1.0F, 
                                                             .min_req_heading_error_to_target = deg2rad(5.0F)};

// reward calculation parameters
// normalized range error reward calculation
static const float normalized_range_error_reward_interp_x1 = 0.01F;
static const float normalized_range_error_reward_interp_y1 = -1.0F;
static const float normalized_range_error_reward_interp_x2 = 0.80F;
static const float normalized_range_error_reward_interp_y2 = -50.0F;

// normalized heading error reward calculation
static const float normalized_heading_error_reward_interp_x1 = 0.01F;
static const float normalized_heading_error_reward_interp_y1 = -1.0F;
static const float normalized_heading_error_reward_interp_x2 = 0.80F;
static const float normalized_heading_error_reward_interp_y2 = -60.0F;

// reward discount factor
static const float discount_factor = 1.0F;

// function definitions

float calc_reward(eig::Array<float, 1, 2, eig::RowMajor>& normalized_policy_state)
{
  // calculate reward for position error
  float reward = RL::linear_interpolate(normalized_policy_state(0, 0), 
                                        normalized_range_error_reward_interp_x1,  normalized_range_error_reward_interp_y1,
                                        normalized_range_error_reward_interp_x2,  normalized_range_error_reward_interp_y2);
  // calculate reward for heading error
  reward += RL::linear_interpolate(fabsf(normalized_policy_state(0, 1)),   
                                   normalized_heading_error_reward_interp_x1,  normalized_heading_error_reward_interp_y1,
                                   normalized_heading_error_reward_interp_x2,  normalized_heading_error_reward_interp_y2);

  return reward;
}


float actor_loss_fcn(const eig::Array<float, eig::Dynamic, 2>& action)
{
  // TODO: Fill this properly, output layer is actions for actor network

  return 0;
}

eig::Array<float, eig::Dynamic, 2>
actor_loss_grad(const eig::Array<float, eig::Dynamic, 2>& action)
{
  // TODO: Fill this properly, output layer is actions for actor network
  return action;
}


float critic_loss_fcn(const eig::Array<float, eig::Dynamic, 1>& G)
{
  return G.mean();
}


eig::Array<float, eig::Dynamic, 1> 
critic_loss_grad(const eig::Array<float, eig::Dynamic, 1>& G)
{
  eig::Array<float, eig::Dynamic, 1> grad(G.rows(), 1);
  grad.fill(1.0F);

  return grad;
}


void learn_to_drive(const RL::GlobalConfig& global_config)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");

  // parameter setup
  constexpr size_t batch_size                  = 256u;
  constexpr size_t max_episodes                = 200u; 
  constexpr size_t warm_up_cycles              = 4u*batch_size;
  constexpr size_t replay_buffer_size          = 20u*batch_size;
  constexpr float  critic_target_smoothing_factor = 0.97F;

  // experience replay setup
  eig::Array<float, eig::Dynamic, BUFFER_LEN, eig::RowMajor> replay_buffer;

  // function approximation setup
  ArtificialNeuralNetwork<2, 8, 12, 20, 25, 2> actor;
  ArtificialNeuralNetwork<4, 10, 14, 21, 27, 1> critic, critic_target;
  AdamOptimizer actor_opt((int)actor.weight.rows(), (int)actor.bias.rows(), 1e-3F);
  AdamOptimizer critic_opt((int)critic.weight.rows(), (int)critic.bias.rows(), 1e-3F);

  actor.dense(Activation(RELU, HE_UNIFORM), 
              Activation(RELU, HE_UNIFORM),
              Activation(RELU, HE_UNIFORM),
              Activation(RELU, HE_UNIFORM),
              Activation(RELU, HE_UNIFORM)
              );
  
  critic.dense(Activation(RELU, HE_UNIFORM), 
               Activation(RELU, HE_UNIFORM),
               Activation(RELU, HE_UNIFORM),
               Activation(RELU, HE_UNIFORM),
               Activation(RELU, HE_UNIFORM) 
               );
  
  // clone critic into target network
  critic_target.weight = critic.weight;
  critic_target.bias = critic.bias;
    
  // random state space sampler for initialization
  std::random_device seed;
  std::mt19937 rand_gen(seed());
  std::uniform_real_distribution<float> state_x_sample(0, world_max_x);
  std::uniform_real_distribution<float> state_y_sample(0, world_max_y);
  std::uniform_real_distribution<float> state_psi_sample(-PI, PI);
  std::uniform_real_distribution<float> action1_sample(-action1_max, action1_max);
  std::uniform_real_distribution<float> action2_sample(-action2_max, action2_max);
  std::normal_distribution<float> action_exploration_dist(0.0F, 1.5F);

  // counter setup
  size_t episode_count = 0u, cycle_count = 0u, replay_buffer_len = 0u;

  while(episode_count < max_episodes)
  {
    RL::DifferentialRobotState cur_state, next_state, target_state;
    eig::Array<float, 1, 2, eig::RowMajor> policy_s_now, policy_action, policy_s_next;
    float reward;
    bool episode_done = false;

    tie(cur_state, target_state) = init_new_episode(state_x_sample, state_y_sample, state_psi_sample, rand_gen);

    // calculate S_t
    tie(policy_s_now(0,0), policy_s_now(0, 1)) = cur_state - target_state;
    state_normalize(global_config, policy_s_now);

    while(NOT(episode_done))
    {
      policy_action = forward_batch<1>(actor, policy_s_now);
      add_exploration_noise(action_exploration_dist, rand_gen, policy_action);

      // clamp policy_action values
      policy_action(0, 0) = std::clamp(policy_action(0, 0), -action1_max, action1_max);
      policy_action(0, 1) = std::clamp(policy_action(0, 1), -action2_max, action2_max);

      next_state  = differential_robot(cur_state, {policy_action(0, 0), policy_action(0, 1)}, global_config);
      next_state.psi = RL::wrapto_minuspi_pi(next_state.psi);

      tie(policy_s_next(0, 0), policy_s_next(0, 1)) = next_state - target_state;
      state_normalize(global_config, policy_s_next);
      reward = calc_reward(policy_s_next);

      episode_done = is_robot_outside_world(next_state, global_config);
      episode_done |= has_robot_reached_target(next_state, target_state, target_reach_params);
      
      // store current transition -> S, A, R, S in replay buffer
      replay_buffer_len %= replay_buffer_size;
      if (replay_buffer.rows() < ((int)replay_buffer_len+1u) )
      { replay_buffer.conservativeResize(replay_buffer_len+1u, NoChange); }
      replay_buffer(replay_buffer_len, {S0, S1})           = policy_s_now;
      replay_buffer(replay_buffer_len, {A0, A1})           = policy_action;
      replay_buffer(replay_buffer_len, R)                  = reward;
      replay_buffer(replay_buffer_len, {NEXT_S0, NEXT_S1}) = policy_s_next;
      replay_buffer(replay_buffer_len, EPISODE_STATE)      = (episode_done == true)?0.0F:1.0F;
      replay_buffer_len++;
      
      if (cycle_count >= warm_up_cycles)
      {
        // sample n measurements of length batch_size
        auto n_transitions = RL::get_n_shuffled_indices<batch_size>((int)replay_buffer.rows());

        eig::Array<float, batch_size, 4, eig::RowMajor> critic_next_input;
        eig::Array<float, batch_size, 1>                Q_next;
        eig::Array<float, batch_size, 1>                Q_now;
        eig::Array<float, batch_size, 1>                td_error;

        // propagate next_state through actor network to calculate A_next = mu(S_next)
        critic_next_input(all, {S0, S1}) = replay_buffer(n_transitions, {NEXT_S0, NEXT_S1});
        critic_next_input(all, {A0, A1}) = forward_batch<batch_size>(actor, critic_next_input(all, {S0, S1}));

        // use citic network with A_next, S_next to calculate Q(S_next, A_next)
        Q_next = forward_batch<batch_size>(critic_target, critic_next_input);

        // use critic network with A_now, S_now to calculate Q(S_now, A_now)
        Q_now = forward_batch<batch_size>(critic, replay_buffer(n_transitions, {S0, S1, A0, A1}));

        // calculate temporal difference = R + gamma*Q(S_next, A_next) - Q(S_now, A_now)
        td_error = replay_buffer(n_transitions, R) + (discount_factor*replay_buffer(n_transitions, EPISODE_STATE)*Q_next) - Q_now;

        // calculate gradient of actor network // TODO: Define this
        auto [actor_loss, actor_weight_grad, actor_bias_grad] = foo();

        // calculate gradient of critic network // TODO: Define this
        auto [critic_loss, critic_weight_grad, critic_bias_grad] = bar();

        // update parameters of actor using optimizer
        actor_opt.step(actor_weight_grad, actor_bias_grad, actor.weight, actor.bias);

        // update parameters of critic using optimizer
        critic_opt.step(critic_weight_grad, critic_bias_grad, critic.weight, critic.bias);

        // soft-update target network
        soft_update_network(critic, critic_target_smoothing_factor, critic_target);
      }
      cur_state     = next_state;
      policy_s_now  = policy_s_next;
      cycle_count   = RL::min(cycle_count++, std::numeric_limits<size_t>::max());
    }

    episode_count++;
  }

  // TODO: return trained actor and critic network (***Last***)
}