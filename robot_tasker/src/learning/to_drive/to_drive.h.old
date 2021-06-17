#ifndef _TO_DRIVE_H_
#define _TO_DRIVE_H_

#include <random>
#include <algorithm>

#include "../global_typedef.h"
#include "../ANN/ANN_typedef.h"
#include "../ANN/ANN.h"
#include "../ANN/ANN_optimizers.h"

#include "../environment_util.h"
#include "../util.h"

#include "robot_dynamics.h"
#include "training_visualizer.h"

using namespace ANN;

float calc_reward(const RL::DifferentialRobotState& state_error)
{
  float pose_error    = std::sqrtf( (state_error.x*state_error.x) + (state_error.y*state_error.y) );
  float heading_error = RL::wrapto_minuspi_pi(state_error.psi);

  // calculate reward for position error
  float reward = RL::linear_interpolate(fabsf(pose_error), 0.1F,  -0.05F, 200.0F,  -2.0F);

  // calculate reward for heading error
  reward += RL::linear_interpolate(fabsf(heading_error),   0.01F, -0.05F, PI, -2.0F);

  return reward;
}


void reward_normalize(float& reward)
{
  reward /= 10.0F;
}


auto calc_error(const RL::DifferentialRobotState& current_state, 
                const RL::DifferentialRobotState& target_state)
{
  auto state_error    = current_state - target_state;
  float pose_error    = std::sqrtf( (state_error.x*state_error.x) + (state_error.y*state_error.y) );
  float heading_error = RL::wrapto_minuspi_pi(state_error.psi);
  return std::make_tuple(pose_error, heading_error);
}


eig::Array<float, 1, 2> state_normalize(const eig::Array<float, 1, 2>& state, 
                                        const RL::GlobalConfig_t& global_config)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float max_pose_error = std::sqrtf(  (world_max_x)*(world_max_x) 
                                                 + (world_max_y)*(world_max_y) );
  eig::Array<float, 1, 2> normalized_state;
  normalized_state(0, 0) = state(0, 0)/max_pose_error;
  normalized_state(0, 1) = state(0, 1)/PI;

  return normalized_state;
}


template<typename EigenDerived>
void action_normalize(const RL::GlobalConfig_t& global_config,
                            eig::ArrayBase<EigenDerived>& action)
{
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");

  action(all, 0) /= (2.0F*action1_max);
  action(all, 1) /= (2.0F*action2_max);
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


auto reset_states(std::uniform_real_distribution<float>& pose_x_sample, 
                  std::uniform_real_distribution<float>& pose_y_sample, 
                  std::uniform_real_distribution<float>& heading_sample, 
                  std::mt19937&                          gen)
{
  RL::DifferentialRobotState initial_state, final_state;
  initial_state.x = pose_x_sample(gen);
  initial_state.y = pose_y_sample(gen);
  initial_state.psi = heading_sample(gen);

  final_state.x = pose_x_sample(gen);
  final_state.y = pose_y_sample(gen);
  final_state.psi = heading_sample(gen);

  return std::make_tuple(initial_state, final_state);
}


float actor_loss(const eig::Array<float, eig::Dynamic, 1>& Q)
{
  float loss = -Q.mean();
  return loss;
}


eig::Array<float, eig::Dynamic, 1>
actor_loss_grad(const eig::Array<float, eig::Dynamic, 1>& Q)
{
  eig::Array<float, eig::Dynamic, 1> retval(Q.rows(), 1);
  retval.fill(-1.0F);

  return retval;
}


float critic_loss(const eig::Array<float, eig::Dynamic, 1>& Q_now, 
                  const eig::Array<float, eig::Dynamic, 1>& Q_next)
{
  float loss = 0.5F*((Q_next - Q_now).square()).mean();
  return loss;
}


template<int BatchSize>
eig::Array<float, BatchSize, 1>
critic_loss_grad(const eig::Array<float, BatchSize, 1>& Q_now, 
                 const eig::Array<float, BatchSize, 1>& Q_next)
{
  // Loss = (0.5/BatchSize)*Sum( Square(Q_target - Q_sampling) )
  eig::Array<float, BatchSize, 1> retval = Q_now - Q_next;
  return retval;
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
  auto [pose_error, heading_error] = calc_error(current_state, target_state);

  if (   (pose_error           < 1.0F)
      && (fabsf(heading_error) < 0.02F) )
  {
    return true;
  } 
  return false;
}                              


template<int BatchSize, 
        int ActorInputSize,  int ... ActorNHiddenLayers, 
        int CriticInputSize, int... CriticNHiddenLayers, 
        typename EigenDerived, typename LossFcn_t, typename LossGradFcn_t>
auto actor_gradient_batch(const ArtificialNeuralNetwork<ActorInputSize, ActorNHiddenLayers ...>&   actor_network,
                          const ArtificialNeuralNetwork<CriticInputSize, CriticNHiddenLayers ...>& critic_network,
                          const eig::ArrayBase<EigenDerived>&                                     input,
                                LossFcn_t&                                                        loss_fcn, 
                                LossGradFcn_t&                                                    loss_grad_fcn, 
                          const RL::GlobalConfig_t&                                               global_config)
{
  constexpr int output_len        = ann_output_len<ActorNHiddenLayers ...>::value;
  constexpr int actor_n_layers    = pack_len<ActorInputSize, ActorNHiddenLayers ...>::value;
  constexpr int critic_n_layers   = pack_len<CriticInputSize, CriticNHiddenLayers ...>::value;
  constexpr int largest_layer_len = max_layer_len<ActorInputSize, ActorNHiddenLayers ..., CriticInputSize, CriticNHiddenLayers ...>::value;
  using weight_type   = decltype(actor_network.weight);
  using bias_type     = decltype(actor_network.bias);
  using Delta_t       = eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor, BatchSize, largest_layer_len>;
  using Activation_t  = eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>;

  const float action1_max = global_config.at("robot/max_wheel_speed");
  const float action2_max = global_config.at("robot/max_wheel_speed");

  tuple<float, weight_type, bias_type> retval = std::make_tuple(0.0F, weight_type{}, bias_type{});
  auto& [loss, weight_grad, bias_grad] = retval;

  // Perform forward propagation
  vector<Activation_t> actor_activations, critic_activations;
  actor_activations.reserve(actor_n_layers);
  critic_activations.reserve(critic_n_layers);
  critic_activations.emplace_back(Activation_t(BatchSize, 4));
  critic_activations[0](all, {0, 1}) = input;
  actor_activations.emplace_back(input);

  // calculate and store activations at each layer for actor_network
  int weights_count = 0;
  int bias_count    = 0;
  for (int layer = 1; layer <= actor_network.n_layers; layer++)
  {
    int n_nodes_last_layer = actor_network.n_nodes(layer-1);
    int n_nodes_cur_layer  = actor_network.n_nodes(layer);
    int weights_start      = weights_count;
    
    auto b = actor_network.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    actor_activations.emplace_back(BatchSize, n_nodes_cur_layer);
    for (int node = 0; node < n_nodes_cur_layer; node++)
    {
      int this_node_weight_start = weights_start+(node*n_nodes_last_layer);
      int this_node_weight_end   = weights_start+((node+1)*n_nodes_last_layer)-1;
      
      // calculate wX + b
      auto w  = actor_network.weight(seq(this_node_weight_start, this_node_weight_end));
      auto wx = (actor_activations[layer-1].matrix() * w.matrix());
      auto z = wx.array() + b(node);
      auto z_normalized = z;//normalize_activations(z);

      actor_activations[layer](all, node) = actor_network.layer_activation_func[layer-1].activation_batch( z_normalized );
    }
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }
  //normalize actor_network actions
  critic_activations[0](all, {2, 3}) = actor_activations.back();
  critic_activations[0](all, 2) = critic_activations[0](all, 2).unaryExpr([](float a){return std::clamp(a, -4.0F, 4.0F); });
  critic_activations[0](all, 3) = critic_activations[0](all, 3).unaryExpr([](float a){return std::clamp(a, -4.0F, 4.0F); });

  // action_normalize(global_config, critic_activations[0](all, {2, 3}));

  // calculate and store activations at each layer for critic_network
  weights_count = 0;
  bias_count    = 0;
  for (int layer = 1; layer <= critic_network.n_layers; layer++)
  {
    int n_nodes_last_layer = critic_network.n_nodes(layer-1);
    int n_nodes_cur_layer  = critic_network.n_nodes(layer);
    int weights_start      = weights_count;
    
    auto b = critic_network.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    critic_activations.emplace_back(BatchSize, n_nodes_cur_layer);
    for (int node = 0; node < n_nodes_cur_layer; node++)
    {
      int this_node_weight_start = weights_start+(node*n_nodes_last_layer);
      int this_node_weight_end   = weights_start+((node+1)*n_nodes_last_layer)-1;
      
      // calculate wX + b
      auto w  = critic_network.weight(seq(this_node_weight_start, this_node_weight_end));
      auto wx = (critic_activations[layer-1].matrix() * w.matrix());
      auto z = wx.array() + b(node);
      auto z_normalized = z;//normalize_activations(z);
      critic_activations[layer](all, node) = critic_network.layer_activation_func[layer-1].activation_batch( z_normalized );
    }
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }

  // calculate loss for the actor
  loss = loss_fcn(critic_activations.back());

  // perform backward propagation to calculate gradient
  // calculate delta for last layer
  Delta_t delta, delta_to_here(loss_grad_fcn(critic_activations.back()));
  
  int n_nodes_this_layer, n_nodes_prev_layer;
  int weight_end, weight_start;
  int bias_end, bias_start;

  int next_layer_weights_end = 0, next_layer_weights_start = (int)critic_network.weight.rows();
  int next_layer_bias_end = 0, next_layer_bias_start = (int)critic_network.bias.rows();
  for (int layer = critic_network.n_layers; layer != 0; layer--)
  {
    n_nodes_this_layer              = critic_network.n_nodes[layer];
    n_nodes_prev_layer              = critic_network.n_nodes[layer-1];
    auto this_layer_activation_grad = critic_network.layer_activation_func[layer-1].grad(critic_activations[layer]);

    // calculate delta
    if (layer != critic_network.n_layers)
    {
      delta.conservativeResize(NoChange, n_nodes_this_layer);
      for (int node = 0; node < n_nodes_this_layer; node++)
      {
        delta(all, node) = (delta_to_here.matrix()*critic_network.weight(seq(next_layer_weights_start+node, 
                                                                             next_layer_weights_end, 
                                                                             n_nodes_this_layer)).eval().matrix());
      }
    }
    else 
    { delta.swap(delta_to_here); }
    delta *= this_layer_activation_grad;
    delta_to_here.swap(delta);

    weight_end   = next_layer_weights_start - 1;
    weight_start = weight_end - (n_nodes_this_layer*n_nodes_prev_layer) + 1;
    bias_end     = next_layer_bias_start - 1;
    bias_start   = bias_end - n_nodes_this_layer + 1;

    next_layer_weights_end   = weight_end;
    next_layer_weights_start = weight_start;
    next_layer_bias_start    = bias_start;
    next_layer_bias_end      = bias_end;
  }

  // gradient for the input layer of critic network
  delta.conservativeResize(NoChange, critic_network.n_nodes[0]);
  for (int node = 0; node < critic_network.n_nodes[0]; node++)
  {
    delta(all, node) = (delta_to_here.matrix()*critic_network.weight(seq(next_layer_weights_start+node, 
                                                                         next_layer_weights_end, 
                                                                         critic_network.n_nodes[0])).eval().matrix());
  }

  // calculate gradient for the actor_network
  delta_to_here = delta(all, {2, 3}); // only take gradient with respect to action_0 and action_1
  next_layer_weights_end = 0, next_layer_weights_start = (int)actor_network.weight.rows();
  next_layer_bias_end = 0, next_layer_bias_start = (int)actor_network.bias.rows();
  for (int layer = actor_network.n_layers; layer != 0; layer--)
  {
    n_nodes_this_layer              = actor_network.n_nodes[layer];
    n_nodes_prev_layer              = actor_network.n_nodes[layer-1];
    auto this_layer_activation_grad = actor_network.layer_activation_func[layer-1].grad(actor_activations[layer]);
    auto& prev_layer_activations    = actor_activations[layer-1];

    // calculate delta
    if (layer != actor_network.n_layers)
    {
      delta.conservativeResize(NoChange, n_nodes_this_layer);
      for (int node = 0; node < n_nodes_this_layer; node++)
      {
        delta(all, node) = (delta_to_here.matrix()*actor_network.weight(seq(next_layer_weights_start+node, 
                                                                            next_layer_weights_end, 
                                                                            n_nodes_this_layer)).eval().matrix());
      }
    }
    else 
    { delta.swap(delta_to_here); }
    delta *= this_layer_activation_grad;

    // calculate gradient
    weight_end   = next_layer_weights_start - 1;
    weight_start = weight_end - (n_nodes_this_layer*n_nodes_prev_layer) + 1;
    bias_end     = next_layer_bias_start - 1;
    bias_start   = bias_end - n_nodes_this_layer + 1;
   for (int node = 0; node < n_nodes_this_layer; node++)
   {
      auto temp = delta(all, node);
      for (int j = 0; j < n_nodes_prev_layer; j++)
      {
        auto weight_grad_list = temp*prev_layer_activations(all, j);
        weight_grad(weight_start+(node*n_nodes_prev_layer)+j) = weight_grad_list.mean();
      }
      bias_grad(bias_start+node) = temp.mean();
   }
    delta_to_here.swap(delta);

    next_layer_weights_end   = weight_end;
    next_layer_weights_start = weight_start;
    next_layer_bias_start    = bias_start;
    next_layer_bias_end      = bias_end;
  }

  return retval;
}


auto learn_to_drive(const RL::GlobalConfig_t& global_config)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");

  // Cppyplot::cppyplot pyp;
  // int temp_payload = 0;
  // pyp.raw(R"pyp(
  // critic_weight_grad_hist = []
  // critic_bias_grad_hist   = []
  // critic_weight_hist      = []
  // critic_bias_hist        = []
  // critic_loss_hist        = []

  // actor_weight_grad_hist  = []
  // actor_bias_grad_hist    = []
  // actor_weight_hist       = []
  // actor_bias_hist         = []
  // actor_loss_hist         = []
  // )pyp", _p(temp_payload));
  // bool start_recording_data   = false;
  // size_t recording_cycle      = 0u;

  const int s0                = 0;
  const int s1                = 1;
  const int a0                = 2;
  const int a1                = 3;
  const int r                 = 4;
  const int next_s0           = 5;
  const int next_s1           = 6;
  const int sim_state         = 7;
  const float discount_factor = 0.45F;

  std::random_device seed;
  std::mt19937 rand_gen(seed());
  std::uniform_real_distribution<float> pose_x_sample(0, world_max_x);
  std::uniform_real_distribution<float> pose_y_sample(0, world_max_y);
  std::uniform_real_distribution<float> heading_sample(-PI, PI);
  std::uniform_real_distribution<float> action1_sample(-action1_max, action1_max);
  std::uniform_real_distribution<float> action2_sample(-action2_max, action2_max);

  constexpr size_t batch_size         = 256u;
  constexpr size_t max_episodes       = 200u;
  constexpr size_t warm_up            = 25000u;
  constexpr size_t replay_buffer_size = 25000u;

  eig::Array<float, eig::Dynamic, 8, eig::RowMajor> replay_buffer;
  replay_buffer.resize(replay_buffer_size, 8);
  size_t replay_buffer_len = 0u;
  size_t episode_count     = 0u; 

  // Deep deterministic policy gradient
  ArtificialNeuralNetwork<2, 8, 12, 20, 25, 2> target_actor; 
  ArtificialNeuralNetwork<4, 10, 14, 21, 27, 1> target_critic;

  target_actor.dense(Activation(RELU, HE_UNIFORM),
                     Activation(RELU, HE_UNIFORM),
                     Activation(RELU, HE_UNIFORM),
                     Activation(RELU, HE_UNIFORM),
                     Activation(RELU, HE_UNIFORM)
                    );
  target_critic.dense(Activation(RELU, HE_UNIFORM), 
                      Activation(RELU, HE_UNIFORM),
                      Activation(RELU, HE_UNIFORM),
                      Activation(RELU, HE_UNIFORM),
                      Activation(RELU, HE_UNIFORM)
                    );

  AdamOptimizer actor_opt((int)target_actor.weight.rows(), (int)target_actor.bias.rows(), 1e-3F);
  AdamOptimizer critic_opt((int)target_critic.weight.rows(), (int)target_critic.bias.rows(), 1e-3F);
  
  size_t current_cycle        = 0u;
  bool initialized_visualizer = false;
  while (episode_count < max_episodes)
  {
    auto [current_state, target_state] = reset_states(pose_x_sample, pose_y_sample, heading_sample, rand_gen);

    eig::Array<float, 1, 2, eig::RowMajor> state;
    eig::Array<float, 1, 2, eig::RowMajor> action;
    float reward;
    eig::Array<float, 1, 2, eig::RowMajor> next_state;
    bool episode_done = false;
    // size_t dbg_counter = 0u;

    while(NOT(episode_done))
    {
      // select action using current state and actor network
      tie(state(0, 0), state(0, 1)) = calc_error(current_state, target_state);
      state  = state_normalize(state, global_config);
      action = forward_batch<1>(target_actor, state);

      // clamp action
      action(0, 0) = std::clamp(action(0, 0), -action1_max, action1_max);
      action(0, 1) = std::clamp(action(0, 1), -action2_max, action2_max);

      // if ( (fabsf(action(0, 0)) > 3.99F) || (fabsf(action(0, 1)) > 3.99F) )
      // {
        // dbg_counter++;
      // }
      // else
      // {
        // dbg_counter = 0u;
      // }

      // if ((dbg_counter > 100u) && (start_recording_data == false))
      // {
      //   start_recording_data = true;
      //   recording_cycle = 0u;
      // }

      // execute the action and observe next state, reward
      auto state_projected = differential_robot(current_state, 
                                                {action(0, 0), action(0, 1)}, 
                                                global_config);
      
      state_projected.psi = RL::wrapto_minuspi_pi(state_projected.psi);

      tie(next_state(0, 0), next_state(0, 1)) = calc_error(state_projected, target_state);
      next_state = state_normalize(next_state, global_config);
      reward     = calc_reward(state_projected - target_state);

      episode_done  = is_robot_outside_world(state_projected, global_config);
      episode_done |= has_robot_reached_target(state_projected, target_state);

      // store current transition -> S_t, A_t, R_t, S_{t+1} in replay buffer
      replay_buffer_len %= replay_buffer_size;
      if (replay_buffer.rows() < (int)(replay_buffer_len+1u))
      { replay_buffer.conservativeResize(replay_buffer_len+1u, NoChange); }
      replay_buffer(replay_buffer_len, {s0, s1})           = state;
      replay_buffer(replay_buffer_len, {a0, a1})           = action;
      replay_buffer(replay_buffer_len, r)                  = reward;
      replay_buffer(replay_buffer_len, {next_s0, next_s1}) = next_state;
      replay_buffer(replay_buffer_len, sim_state)          = (episode_done == true)? 0.0F:1.0F;
      replay_buffer_len++;

      current_state = state_projected;

      // sample n-different actions using the current state and store it in the replay buffer
      for (size_t n = 0u; n < 8u; n++)
      {
        tie(state(0, 0), state(0, 1)) = calc_error(current_state, target_state);
        state        = state_normalize(state, global_config);
        action(0, 0) = action1_sample(rand_gen);
        action(0, 1) = action2_sample(rand_gen);

        state_projected = differential_robot(current_state, 
                                            {action(0, 0), action(0, 1)}, 
                                            global_config);
      
        state_projected.psi = RL::wrapto_minuspi_pi(state_projected.psi);

        tie(next_state(0, 0), next_state(0, 1)) = calc_error(state_projected, target_state);
        next_state = state_normalize(next_state, global_config);
        reward     = calc_reward(state_projected - target_state);

        // store in replay buffer
        // store current transition -> S_t, A_t, R_t, S_{t+1} in replay buffer
        replay_buffer_len %= replay_buffer_size;
        if (replay_buffer.rows() < (int)(replay_buffer_len+1u))
        { replay_buffer.conservativeResize(replay_buffer_len+1u, NoChange); }
        replay_buffer(replay_buffer_len, {s0, s1})           = state;
        replay_buffer(replay_buffer_len, {a0, a1})           = action;
        replay_buffer(replay_buffer_len, r)                  = reward;
        replay_buffer(replay_buffer_len, {next_s0, next_s1}) = next_state;
        replay_buffer(replay_buffer_len, sim_state)          = (episode_done == true)? 0.0F:1.0F;
        replay_buffer_len++;
      }
      current_cycle++;

      if (current_cycle > warm_up)
      {
        if (NOT(initialized_visualizer))
        {
          training_visualizer_init();
          initialized_visualizer = true;
        }

        // Sample 'N' transitions from replay buffer to do mini-batch param optimization
        auto n_transitions = get_n_shuffled_idx<batch_size>((int)replay_buffer.rows());

        eig::Array<float, batch_size, 1> Q_next;
        eig::Array<float, batch_size, 4, eig::RowMajor> target_critic_input;

        target_critic_input(all, {s0, s1}) = replay_buffer(n_transitions, {next_s0, next_s1});
        target_critic_input(all, {a0, a1}) = forward_batch<batch_size>(target_actor, target_critic_input(all, {s0, s1}));
        
        // normalize actions
        // action_normalize(global_config, target_critic_input(all, {a0, a1}));

        Q_next  = forward_batch<batch_size>(target_critic, target_critic_input);
        Q_next *= (discount_factor*replay_buffer(n_transitions, sim_state));
        Q_next += replay_buffer(n_transitions, r);

        // calculate loss between Q_target, Q_sampling and perform optimization step
        auto [loss_critic, critic_weight_grad, critic_bias_grad] = gradient_batch<batch_size>(target_critic,
                                                                                              replay_buffer(n_transitions, {s0, s1, a0, a1}),
                                                                                              Q_next,
                                                                                              critic_loss,
                                                                                              critic_loss_grad<batch_size>);
        critic_opt.step(critic_weight_grad, critic_bias_grad, target_critic.weight, target_critic.bias);

        // calculate loss for sampling_actor network and perform optimization step
        auto [loss_actor, actor_weight_grad, actor_bias_grad] = actor_gradient_batch<batch_size>(target_actor,
                                                                                                 target_critic, 
                                                                                                 replay_buffer(n_transitions, {s0, s1}),
                                                                                                 actor_loss,
                                                                                                 actor_loss_grad,
                                                                                                 global_config);
        actor_opt.step(actor_weight_grad, actor_bias_grad, target_actor.weight, target_actor.bias);

        // if (start_recording_data == true)
        // {
        //   auto& critic_weight = target_critic.weight;
        //   auto& critic_bias   = target_critic.bias;
        //   auto& actor_weight  = target_actor.weight;
        //   auto& actor_bias    = target_actor.bias;

        //   pyp.raw(R"pyp(
        //   critic_weight_grad_hist.append(critic_weight_grad)
        //   critic_bias_grad_hist.append(critic_bias_grad)
        //   critic_weight_hist.append(critic_weight)
        //   critic_bias_hist.append(critic_bias)
        //   critic_loss_hist.append(loss_critic)

        //   actor_weight_grad_hist.append(actor_weight_grad)
        //   actor_bias_grad_hist.append(actor_bias_grad)
        //   actor_weight_hist.append(actor_weight)
        //   actor_bias_hist.append(actor_bias)
        //   actor_loss_hist.append(loss_actor)
        //   )pyp", _p(critic_weight_grad), _p(critic_bias_grad), _p(critic_weight), _p(critic_bias), _p(loss_critic), 
        //          _p(actor_weight_grad),  _p(actor_bias_grad),  _p(actor_weight),  _p(actor_bias),  _p(loss_actor));
        //   recording_cycle++;
        // }
        
        if (current_cycle % 5 == 0)
        { training_visualizer_update(loss_critic, loss_actor, {action(0, 0), action(0, 1)}, Q_next.mean()); }

        // if (recording_cycle >= 100)
        // { episode_count = 500; break; }

        std::cout << "Episode: " << episode_count << ", Cycle: " << current_cycle << ", LossCritic: " << loss_critic << ", LossActor: " << loss_actor << '\n';
      }
    }
    episode_count++;
  }

  // pyp.raw(R"pyp(
  // np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/critic_weight_grad_hist.npy", np.array(critic_weight_grad_hist))
  // np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/critic_bias_grad_hist.npy", np.array(critic_bias_grad_hist))
  // np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/critic_weight_hist.npy", np.array(critic_weight_hist))
  // np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/critic_bias_hist.npy", np.array(critic_bias_hist))
  // np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/critic_loss_hist.npy", np.array(critic_loss_hist))

  // np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/actor_weight_grad_hist.npy", np.array(actor_weight_grad_hist))
  // np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/actor_bias_grad_hist.npy", np.array(actor_bias_grad_hist))
  // np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/actor_weight_hist.npy", np.array(actor_weight_hist))
  // np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/actor_bias_hist.npy", np.array(actor_bias_hist))
  // np.save("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/actor_loss_hist.npy", np.array(actor_loss_hist))
  // )pyp", _p(temp_payload));

  return std::make_tuple(target_actor, target_critic);
}


#endif
