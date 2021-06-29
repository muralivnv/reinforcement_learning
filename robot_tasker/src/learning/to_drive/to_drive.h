#include <Eigen/Core>
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

#include "../../util/environment_util.h"

using namespace ANN;
using namespace RL;

// local parameters
static const TargetReachSuccessParams target_reach_params = TargetReachSuccessParams{1.0F, deg2rad(5.0F)};

// reward calculation parameters
// normalized range error reward calculation
static const float normalized_range_error_reward_interp_x1 = 0.01F;
static const float normalized_range_error_reward_interp_y1 = -0.1F; // -1.0F
static const float normalized_range_error_reward_interp_x2 = 0.80F;
static const float normalized_range_error_reward_interp_y2 = -2.0F; // -5.0F

// normalized heading error reward calculation
static const float normalized_heading_error_reward_interp_x1 = 0.01F;
static const float normalized_heading_error_reward_interp_y1 = -0.1F; // -1.0F
static const float normalized_heading_error_reward_interp_x2 = 0.80F;
static const float normalized_heading_error_reward_interp_y2 = -4.0F; // -6.0F

// reward discount factor
static const float discount_factor = 0.7F;

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


void learn_to_drive(const RL::GlobalConfig_t& global_config)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");

  // parameter setup
  constexpr size_t batch_size                  = 256u;
  const size_t max_episodes                = 200u; 
  const size_t warm_up_cycles              = 4u*batch_size;
  const size_t replay_buffer_size          = 20u*batch_size;
  const float  critic_target_smoothing_factor = 0.999F;
  const float gradient_norm_threshold         = 0.5F;
  const size_t critic_target_update_rate      = 500u;
  const float  actor_l2_reg_factor            = 1e-2F;
  const float  critic_l2_reg_factor           = 1e-3F;

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
  // std::normal_distribution<float> action_exploration_dist(0.0F, 6.5F);

  // counter setup
  size_t episode_count = 0u, cycle_count = 0u, replay_buffer_len = 0u;

  // debug initialization --start
  // visualizer specific variables
  Cppyplot::cppyplot pyp;
  bool vis_initialized  = false;
  bool vis_episode_done = false;
  RL::DifferentialRobotState vis_cur_state, vis_target_state, vis_next_state;
  eig::Array<float, 1, 2, eig::RowMajor> vis_policy_s_now, vis_policy_action, vis_policy_s_next;
  // float vis_Q;

  // data recording for analysis 
  bool record_data = false;
  bool initialized_data_record = false;
  size_t max_cycles_to_record = 2000;

  // debug initialization --end

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
      // policy_action = forward_batch<1>(actor, policy_s_now);
      // float exploration_noise = get_exploration_noise(action_exploration_dist, rand_gen);
      // exploration_noise /= action1_max;
      // policy_action += exploration_noise;
      // policy_action(0, 0) = std::clamp(policy_action(0, 0), 0.0F, 1.0F);
      // policy_action(0, 1) = std::clamp(policy_action(0, 1), 0.0F, 1.0F);
      cur_state.x = state_x_sample(rand_gen);
      cur_state.y = state_y_sample(rand_gen);
      cur_state.psi = state_psi_sample(rand_gen);

      tie(policy_s_now(0, 0), policy_s_now(0, 1)) = cur_state - target_state;
      state_normalize(global_config, policy_s_now);
      policy_action(0, 0) = action1_sample(rand_gen)/action1_max;
      policy_action(0, 1) = action2_sample(rand_gen)/action2_max;

      next_state  = differential_robot(cur_state, {policy_action(0, 0)*action1_max, policy_action(0, 1)*action2_max}, global_config);
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

      //debug code --start
      if (record_data == true)
      {
        if (cycle_count < max_cycles_to_record)
        {
          if (initialized_data_record == false)
          {
            pyp.raw(R"pyp(
            replay_buffer = [None]*max_cycles_to_record
            buffer_idx = 0
            )pyp", _p(max_cycles_to_record));
            initialized_data_record = true;
          }

          const float s1 = policy_s_now(0, 0), s2 = policy_s_now(0, 1);
          const float a1 = policy_action(0, 0), a2 = policy_action(0, 1);
          const float s1_next = policy_s_next(0, 0), s2_next = policy_s_next(0, 1);
          pyp.raw(R"pyp(
          replay_buffer[buffer_idx] = [s1, s2, a1, a2, reward, s1_next, s2_next]
          buffer_idx += 1
          )pyp", _p(s1), _p(s2), _p(a1), _p(a2), _p(reward), _p(s1_next), _p(s2_next));
        }
        else
        {
          int temp_payload = 0;
          pyp.raw(R"pyp(
          np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/scripts/analysis/replay_buffer.npy", np.array(replay_buffer))
          plt.plot([], [])
          plt.show()
          )pyp", _p(temp_payload));
          std::cout << "Saved replay buffer for further analysis\n";

          record_data = false;
        }
      }
      //debug code --end

      if (cycle_count >= warm_up_cycles)
      {
        // sample n measurements of length batch_size
        auto n_transitions = RL::get_n_shuffled_indices<batch_size>((int)replay_buffer.rows());

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

        // calculate gradient of actor network
        auto [actor_loss, actor_weight_grad, actor_bias_grad] = actor_gradient_batch<batch_size>(actor, 
                                                                                                 critic, 
                                                                                                 replay_buffer(n_transitions, {S0, S1}),
                                                                                                 actor_loss_fcn,
                                                                                                 actor_loss_grad, 
                                                                                                 global_config);
        // calculate gradient of critic network
        auto [critic_loss, critic_weight_grad, critic_bias_grad] = gradient_batch<batch_size>(critic, 
                                                                                              replay_buffer(n_transitions, {S0, S1, A0, A1}),
                                                                                              td_error,
                                                                                              critic_loss_fcn,
                                                                                              critic_loss_grad);

        // Add L2 regularization for both actor and critic network
        actor_weight_grad  += (actor_l2_reg_factor*actor.weight)/(float)batch_size;
        critic_weight_grad += (critic_l2_reg_factor*critic.weight)/(float)batch_size;

        //debug code --start
        float actor_weight_grad_norm  = std::sqrtf(actor_weight_grad.square().sum());
        float actor_bias_grad_norm    = std::sqrtf(actor_bias_grad.square().sum());
        float critic_weight_grad_norm = std::sqrtf(critic_weight_grad.square().sum());
        float critic_bias_grad_norm   = std::sqrtf(critic_bias_grad.square().sum());
        //debug code --end

        // update parameters of actor using optimizer
        actor_opt.step(actor_weight_grad, actor_bias_grad, actor.weight, actor.bias);

        // update parameters of critic using optimizer
        critic_opt.step(critic_weight_grad, critic_bias_grad, critic.weight, critic.bias);

        // hard-update critic target network
        if ( (cycle_count % critic_target_update_rate) == 0u)
        {
          critic_target.weight = critic.weight;
          critic_target.bias   = critic.bias;
          std::cout << "C: " << cycle_count << " | updated critic target\n" << std::flush;
        }

        //debug code --start
        if ( (cycle_count % 20) == 0)
        {
          std::cout << "E: " << episode_count << " | C: " << cycle_count
                    << " | Ac: " << actor_loss << " | Cr: " << critic_loss 
                    << " | Ac_WNrm: " << actor_weight_grad_norm << " | Ac_BNrm: " << actor_bias_grad_norm
                    << " | Cr_WNrm: " << critic_weight_grad_norm << " | Cr_BNrm: " << critic_bias_grad_norm << '\n';
          std::cout << std::flush;
        }

        if ( (cycle_count >= 2000) && ( (cycle_count % 20) == 0) )
        {
          std::cout << "C: " << cycle_count << " | Ac -> ";
          display_layer_weights_norm(actor);
          std::cout << std::flush;

          std::cout << "C: " << cycle_count << " | Cr -> ";
          display_layer_weights_norm(critic);
          std::cout << std::flush;
        }
      }
      //debug code --end

      cur_state     = next_state;
      policy_s_now  = policy_s_next;
      cycle_count   = RL::min(++cycle_count, std::numeric_limits<size_t>::max());
      
      //debug code --start
      // VISUALIZATION START
      if ( (cycle_count >= 2000u) && ((cycle_count % 10) == 0u) )
      {
        if (NOT(vis_initialized))
        {
          ENV::realtime_visualizer_init("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/world_barriers.csv", 
                                        10);
          vis_initialized = true;
          tie(vis_cur_state, vis_target_state) = init_new_episode(state_x_sample, state_y_sample, state_psi_sample, rand_gen);
          ENV::update_target_pose({vis_target_state.x, vis_target_state.y});
        }

        if (vis_episode_done == true)
        {
          tie(vis_cur_state, vis_target_state) = init_new_episode(state_x_sample, state_y_sample, state_psi_sample, rand_gen);
          ENV::update_target_pose({vis_target_state.x, vis_target_state.y});
        }
        
        tie(vis_policy_s_now(0,0), vis_policy_s_now(0, 1)) = vis_cur_state - vis_target_state;
        state_normalize(global_config, vis_policy_s_now);
        
        vis_policy_action = forward_batch<1>(actor, vis_policy_s_now);
        
        vis_next_state  = differential_robot(vis_cur_state, {vis_policy_action(0, 0)*action1_max, vis_policy_action(0, 1)*action2_max}, global_config);
        vis_next_state.psi = RL::wrapto_minuspi_pi(vis_next_state.psi);
        
        vis_cur_state = vis_next_state;

        vis_episode_done = is_robot_outside_world(vis_next_state, global_config);
        vis_episode_done |= has_robot_reached_target(vis_next_state, vis_target_state, target_reach_params);

        ENV::update_visualizer({vis_next_state.x, vis_next_state.y}, 
                              {vis_policy_action(0,0)*action1_max, vis_policy_action(0,1)*action2_max}, 
                              0.0F, 10);
      }
      //debug code --end
    }

    episode_count++;
  }

  // TODO: return trained actor and critic network (***Last***)
}