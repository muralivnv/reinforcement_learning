#ifndef _TO_DRIVE_H_
#define _TO_DRIVE_H_

#include <iostream>
#include <limits>
#include <random>
#include <cmath>

#include "../../global_typedef.h"

#include "../../ANN/ANN_activation.h"
#include "../../ANN/ANN.h"
#include "../../ANN/ANN_optimizers.h"
#include "../../ANN/ANN_util.h"

#include "../../util/util.h"

#include "robot_dynamics.h"
#include "to_drive_util.h"

using namespace ANN;

namespace learning::to_drive
{

// local parameters
static const TargetReachSuccessParams target_reach_params = TargetReachSuccessParams{1.0F, deg2rad(5.0F)};

// reward calculation parameters
// normalized range error reward calculation
static const float normalized_range_error_reward_interp_x1 = 0.01F;
static const float normalized_range_error_reward_interp_y1 = -0.1F;
static const float normalized_range_error_reward_interp_x2 = 0.80F;
static const float normalized_range_error_reward_interp_y2 = -2.0F;

// normalized heading error reward calculation
static const float normalized_heading_error_reward_interp_x1 = 0.01F;
static const float normalized_heading_error_reward_interp_y1 = -0.1F;
static const float normalized_heading_error_reward_interp_x2 = 0.80F;
static const float normalized_heading_error_reward_interp_y2 = -4.0F;

// reward discount factor
static const float discount_factor = 0.7F;

// function definitions
float calc_reward(eig::Array<float, 1, 2, eig::RowMajor>& normalized_policy_state)
{
  // calculate reward for position error
  float reward = util::linear_interpolate(normalized_policy_state(0, 0), 
                                        normalized_range_error_reward_interp_x1,  normalized_range_error_reward_interp_y1,
                                        normalized_range_error_reward_interp_x2,  normalized_range_error_reward_interp_y2);
  // calculate reward for heading error
  reward += util::linear_interpolate(fabsf(normalized_policy_state(0, 1)),   
                                   normalized_heading_error_reward_interp_x1,  normalized_heading_error_reward_interp_y1,
                                   normalized_heading_error_reward_interp_x2,  normalized_heading_error_reward_interp_y2);

  return reward;
}


float actor_loss_fcn(const eig::Array<float, eig::Dynamic, 1>& Q)
{
  // J_actor = -(1/N)* summation (Q)
  float loss = -(Q.mean());
  return loss;
}


eig::Array<float, eig::Dynamic, 1>
actor_loss_grad(const eig::Array<float, eig::Dynamic, 1>& Q)
{
  eig::Array<float, eig::Dynamic, 1> grad(Q.rows(), 1);
  grad.fill(-1.0F);
  return grad;
}


float critic_loss_fcn(const eig::Array<float, eig::Dynamic, 1>& Q, 
                      const eig::Array<float, eig::Dynamic, 1>& td_error)
{
  UNUSED(Q);
  float loss = 0.5F * (td_error.square()).mean();
  return loss;
}


eig::Array<float, eig::Dynamic, 1> 
critic_loss_grad(const eig::Array<float, eig::Dynamic, 1>& Q, 
                 const eig::Array<float, eig::Dynamic, 1>& td_error)
{
  UNUSED(Q);
  eig::Array<float, eig::Dynamic, 1> grad = -td_error;
  return grad;
}


auto learn_to_drive(const learning::to_drive::GlobalConfig_t& global_config, const bool logging_enabled = true)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");

  // parameter setup
  constexpr size_t batch_size = 256u;
  const size_t max_episodes = 200u; 
  const size_t warm_up_cycles = 4u*batch_size;
  const size_t replay_buffer_size = 20u*batch_size;
  const size_t critic_target_update_ncycles = 500u;
  const float  actor_l2_reg_factor = 1e-2F;
  const float  critic_l2_reg_factor = 1e-3F;

  // experience replay setup
  eig::Array<float, eig::Dynamic, BUFFER_LEN, eig::RowMajor> replay_buffer;

  // function approximation setup
  ArtificialNeuralNetwork<2, 8, 12, 20, 25, 2> actor;
  ArtificialNeuralNetwork<4, 10, 14, 21, 27, 1> critic, critic_target;
  AdamOptimizer actor_opt((int)actor.weight.rows(), (int)actor.bias.rows(), 1e-4F);
  AdamOptimizer critic_opt((int)critic.weight.rows(), (int)critic.bias.rows(), 1e-4F);

  actor.dense(Activation(RELU, HE_UNIFORM), 
              Activation(RELU, HE_UNIFORM),
              Activation(RELU, HE_UNIFORM),
              Activation(RELU, HE_UNIFORM),
              Activation(SIGMOID, HE_UNIFORM)
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
  std::uniform_real_distribution<float> state_x_sample(0.0F, world_max_x);
  std::uniform_real_distribution<float> state_y_sample(0.0F, world_max_y);
  std::uniform_real_distribution<float> state_psi_sample(-PI, PI);
  std::uniform_real_distribution<float> action1_sample(0.0F, action1_max);
  std::uniform_real_distribution<float> action2_sample(0.0F, action2_max);

  // counter setup
  size_t episode_count = 0u, cycle_count = 0u, replay_buffer_len = 0u;
  float critic_loss_avg = 0.0F, actor_loss_avg = 0.0F;
  float loss_smoothing_factor = 0.90F;
  bool terminate_actor_optim = false;
  bool terminate_critic_optim = false;
  size_t critic_optim_termination_counter = 0u;
  size_t actor_optim_termination_counter  = 0u;

  while(   (episode_count < max_episodes        ) 
        && (   (terminate_actor_optim == false ) 
            || (terminate_critic_optim == false)) )
  {
    DifferentialRobotState cur_state, next_state, target_state;
    eig::Array<float, 1, 2, eig::RowMajor> policy_s_now, policy_action, policy_s_next;
    float reward;
    bool episode_done = false;

    tie(cur_state, target_state) = init_new_episode(state_x_sample, state_y_sample, state_psi_sample, rand_gen);

    while(NOT(episode_done))
    {
      // sample random robot state from uniform distribution (for better exploration)
      cur_state.x = state_x_sample(rand_gen);
      cur_state.y = state_y_sample(rand_gen);
      cur_state.psi = state_psi_sample(rand_gen);
      
      // calculate policy state
      tie(policy_s_now(0, 0), policy_s_now(0, 1)) = cur_state - target_state;
      state_normalize(global_config, policy_s_now);

      // sample random robot state from uniform distribution (for better exploration)
      policy_action(0, 0) = action1_sample(rand_gen)/action1_max;
      policy_action(0, 1) = action2_sample(rand_gen)/action2_max;

      // perturb system with sample state and actions to observe next state and reward
      next_state  = differential_robot(cur_state, {policy_action(0, 0)*action1_max, policy_action(0, 1)*action2_max}, global_config);
      next_state.psi = util::wrapto_minuspi_pi(next_state.psi);

      tie(policy_s_next(0, 0), policy_s_next(0, 1)) = next_state - target_state;
      state_normalize(global_config, policy_s_next);
      reward = calc_reward(policy_s_next);

      // check whether a reset is required or not
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
        auto n_transitions = util::get_n_shuffled_indices<batch_size>((int)replay_buffer.rows());
        const float actor_loss_prev = actor_loss_avg;
        const float critic_loss_prev = critic_loss_avg;

        if (NOT(terminate_critic_optim))
        {
          eig::Array<float, batch_size, 4, eig::RowMajor> critic_next_input;
          eig::Array<float, batch_size, 1>                Q_next, Q_now, td_error;

          // propagate next_state through actor network to calculate A_next = mu(S_next)
          critic_next_input(all, {S0, S1}) = replay_buffer(n_transitions, {NEXT_S0, NEXT_S1});
          critic_next_input(all, {A0, A1}) = forward_batch<batch_size>(actor, critic_next_input(all, {S0, S1}));

          // use citic network with A_next, S_next to calculate Q(S_next, A_next)
          Q_next = forward_batch<batch_size>(critic_target, critic_next_input);

          // use critic network with A_now, S_now to calculate Q(S_now, A_now)
          Q_now = forward_batch<batch_size>(critic, replay_buffer(n_transitions, {S0, S1, A0, A1}));

          // calculate temporal difference = R + gamma*Q(S_next, A_next) - Q(S_now, A_now)
          td_error = replay_buffer(n_transitions, (int)R) + (discount_factor*replay_buffer(n_transitions, (int)EPISODE_STATE)*Q_next) - Q_now;

          // calculate gradient of critic network
          auto [critic_loss, critic_weight_grad, critic_bias_grad] = gradient_batch<batch_size>(critic, 
                                                                                                replay_buffer(n_transitions, {S0, S1, A0, A1}),
                                                                                                td_error,
                                                                                                critic_loss_fcn,
                                                                                                critic_loss_grad);
          critic_weight_grad += (critic_l2_reg_factor*critic.weight)/(float)batch_size;

          // update parameters of critic using optimizer
          critic_opt.step(critic_weight_grad, critic_bias_grad, critic.weight, critic.bias);

          // hard-update critic target network
          if ( (cycle_count % critic_target_update_ncycles) == 0u)
          {
            critic_target.weight = critic.weight;
            critic_target.bias   = critic.bias;
            if (logging_enabled == true)
            { 
              std::cout << "C: " << cycle_count << " | updated critic target\n" << std::flush;
            }
          }

          critic_loss_avg *= loss_smoothing_factor;
          critic_loss_avg += (1.0F - loss_smoothing_factor)*critic_loss;
        }

        if (NOT(terminate_actor_optim))
        {
          // calculate gradient of actor network
          auto [actor_loss, actor_weight_grad, actor_bias_grad] = actor_gradient_batch<batch_size>(actor, 
                                                                                                  critic, 
                                                                                                  replay_buffer(n_transitions, {S0, S1}),
                                                                                                  actor_loss_fcn,
                                                                                                  actor_loss_grad, 
                                                                                                  global_config);
          // Use L2 regularization of weights for both actor and critic network
          actor_weight_grad  += (actor_l2_reg_factor*actor.weight)/(float)batch_size;
        

          // update parameters of actor using optimizer
          actor_opt.step(actor_weight_grad, actor_bias_grad, actor.weight, actor.bias);

          // update average loss
          actor_loss_avg *= loss_smoothing_factor;
          actor_loss_avg += (1.0F - loss_smoothing_factor)*actor_loss;
        }

        if (logging_enabled == true)
        {
          if ( (cycle_count % 20) == 0)
          {
            std::cout << "E: " << episode_count << " | C: " << cycle_count
                      << " | AcL: " << actor_loss_avg << " | CrL: " << critic_loss_avg
                      << " | AcLDelChng: " << fabsf(actor_loss_prev - actor_loss_avg)
                      << " | CrLDelChng: " << fabsf(critic_loss_prev - critic_loss_avg) << '\n';
            std::cout << std::flush;
          }
        }

        if (fabsf(critic_loss_avg) < 3e-3F)
        {
          critic_optim_termination_counter++;
          if (critic_optim_termination_counter >= critic_target_update_ncycles)
          {
            terminate_critic_optim = true;
          }
        }
        else
        {
          critic_optim_termination_counter = 0u;
        }

        if (terminate_critic_optim == true)
        {
          actor_optim_termination_counter++;
          if (actor_optim_termination_counter >= (2u*batch_size) )
          {
            terminate_actor_optim = true;
            break;
          }
        }
      }
      cycle_count   = util::min(++cycle_count, std::numeric_limits<size_t>::max());
    }

    episode_count++;
  }

  std::cout << "I know how to Drive!\n";
  return std::make_tuple(actor, critic);
}

} // namespace {learning::to_drive}

#endif