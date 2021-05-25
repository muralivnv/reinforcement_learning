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

using namespace ANN;

// local parameters
static const float min_req_range_error_to_target   = 1.0F;
static const float min_req_heading_error_to_target = deg2rad(5.0F); 

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
static const float discount_factor = 0.75F;

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

tuple<float, float> operator-(const RL::DifferentialRobotState& lhs, 
                              const RL::DifferentialRobotState& rhs)
{
  float range_error, heading_error;
  range_error = std::sqrtf( RL::squaref(lhs.x - rhs.x) + RL::squaref(lhs.y - rhs.y) );
  heading_error = RL::wrapto_minuspi_pi(lhs.psi - rhs.psi);

  return std::make_tuple(range_error, heading_error);
}

tuple<RL::DifferentialRobotState, RL::DifferentialRobotState>
init_new_episode(std::uniform_real_distribution<float>& state_x_sample, 
                 std::uniform_real_distribution<float>& state_y_sample, 
                 std::uniform_real_distribution<float>& state_psi_sample, 
                 std::mt19937& rand_gen)
{
  tuple ret_val = std::make_tuple(RL::DifferentialRobotState(), DifferentialRobotState());
  auto& [init_state, final_state] = ret_val;
  
  init_state.x   = state_x_sample(rand_gen);
  init_state.y   = state_y_sample(rand_gen);
  init_state.psi = state_psi_sample(rand_gen);

  final_state.x   = state_x_sample(rand_gen);
  final_state.y   = state_y_sample(rand_gen);
  final_state.psi = state_psi_sample(rand_gen);

  return ret_val;
}

void state_normalize(const RL::GlobalConfig_t&               global_config, 
                     eig::Array<float, 1, 2, eig::RowMajor>& policy_state)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float max_range_error = std::sqrtf(  (world_max_x)*(world_max_x) 
                                                  + (world_max_y)*(world_max_y) );
  static const float max_heading_error = PI;

  policy_state(0, 0) /= max_pose_error;
  policy_state(0, 1) /= max_heading_error;
}


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


bool is_robot_outside_world(const RL::DifferentialRobotState& state,
                            const RL::GlobalConfig_t&         global_config)
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

bool has_robot_reached_target(const RL::DifferentialRobotState& current_state, 
                              const RL::DifferentialRobotState& target_state)
{
  bool is_reached = false;
  auto [range_error, heading_error] = current_state - target_state;

  if (   (range_error          < min_req_range_error_to_target  )
      && (fabsf(heading_error) < min_req_heading_error_to_target) )
  {
    is_reached = true;
  }
  return is_reached;
}


void learn_to_drive(const RL::GlobalConfig& global_config)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");

  constexpr size_t batch_size         = 256u;
  constexpr size_t max_episodes       = 200u; 
  constexpr size_t warm_up_cycles     = 4u*batch_size;
  constexpr size_t replay_buffer_size = 10u*batch_size;
  size_t replay_buffer_len = 0u;

  eig::Array<float, eig::Dynamic, BUFFER_LEN, eig::RowMajor> replay_buffer;
  ArtificialNeuralNetwork<2, 8, 12, 20, 25, 2> actor;
  ArtificialNeuralNetwork<4, 10, 14, 21, 27, 1> critic;
  AdamOptimizer actor_opt((int)actor.weight.rows(), (int)actor.bias.rows(), 1e-3F);
  AdamOptimizer critic_opt((int)critic.weight.rows(), (int)critic.bias.rows(), 1e-3F);
  size_t episode_count = 0u, cycle_count = 0u;

  // random state space sampler for initialization
  std::random_device seed;
  std::mt19937 rand_gen(seed());
  std::uniform_real_distribution<float> state_x_sample(0, world_max_x);
  std::uniform_real_distribution<float> state_y_sample(0, world_max_y);
  std::uniform_real_distribution<float> state_psi_sample(-PI, PI);
  std::uniform_real_distribution<float> action1_sample(-action1_max, action1_max);
  std::uniform_real_distribution<float> action2_sample(-action2_max, action2_max);

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

      // clamp policy_action values
      policy_action(0, 0) = std::clamp(policy_action(0, 0), -action1_max, action1_max);
      policy_action(0, 1) = std::clamp(policy_action(0, 1), -action2_max, action2_max);

      next_state  = differential_robot(cur_state, {policy_action(0, 0), policy_action(0, 1)}, global_config);
      next_state.psi = RL::wrapto_minuspi_pi(next_state.psi);

      tie(policy_s_next(0, 0), policy_s_next(0, 1)) = next_state - target_state;
      state_normalize(global_config, policy_s_next);
      reward = calc_reward(policy_s_next);

      episode_done = is_robot_outside_world(next_state, global_config);
      episode_done |= has_robot_reached_target(policy_s_next);
      
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
        Q_next = forward_batch<batch_size>(critic, critic_next_input);

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
      }
      cur_state     = next_state;
      policy_s_now  = policy_s_next;
      cycle_count   = RL::min(cycle_count++, std::numeric_limits<size_t>::max());
    }

    episode_count++;
  }

  // TODO: return trained actor and critic network
}