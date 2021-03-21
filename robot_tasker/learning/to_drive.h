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

using namespace ANN;

float calc_reward(const RL::DifferentialRobotState& state_error)
{
  float pose_error    = std::sqrtf( (state_error.x*state_error.x) + (state_error.y*state_error.y) );
  float heading_error = RL::wrapto_minuspi_pi(state_error.psi);

  // calculate reward for position error
  float reward = RL::linear_interpolate(fabsf(pose_error), 0.1F,  -0.05F, 200.0F,  -25.0F);

  // calculate reward for heading error
  reward += RL::linear_interpolate(fabsf(heading_error),   0.01F, -0.05F, PI, -15.0F);

  return reward;
}


auto calc_error(const RL::DifferentialRobotState& current_state, 
                const RL::DifferentialRobotState& target_state)
{
  auto state_error    = current_state - target_state;
  float pose_error    = std::sqrtf( (state_error.x*state_error.x) + (state_error.y*state_error.y) );
  float heading_error = RL::wrapto_minuspi_pi(state_error.psi);

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

float actor_loss(const eig::Array<float, eig::Dynamic, 1>& reward)
{
  float loss = -reward.mean();

  return loss;
}

eig::Array<float, eig::Dynamic, 1>
actor_loss_grad(const eig::Array<float, eig::Dynamic, 1>& reward)
{
  eig::Array<float, eig::Dynamic, 1> retval(reward.rows(), 1);
  retval = reward.unaryExpr([](float v){return (v < 0.0F)?1.0F:-1.0F;});

  return retval;
}

float critic_loss(const eig::Array<float, eig::Dynamic, 1>& Q_sampling, 
                  const eig::Array<float, eig::Dynamic, 1>& Q_target)
{
  float loss = 0.5F*((Q_sampling - Q_target).square()).mean();
  return loss;
}

template<int BatchSize>
eig::Array<float, BatchSize, 1>
critic_loss_grad(const eig::Array<float, BatchSize, 1>& Q_sampling, 
                 const eig::Array<float, BatchSize, 1>& Q_target)
{
  // Loss = (0.5/BatchSize)*Sum( Square(Q_sampling - Q_target) )
  eig::Array<float, BatchSize, 1> retval = (Q_sampling - Q_target);
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


eig::Array<float, 1, 2> state_normalize(const eig::Array<float, 1, 2>& state, 
                                        const RL::GlobalConfig_t& global_config)
{
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float max_pose_error = std::sqrtf(  (world_max_x)*(world_max_x) 
                                                 + (world_max_y)*(world_max_y) );
  eig::Array<float, 1, 2> normalized_state;
  normalized_state(0, 0) = state(0, 0)/max_pose_error;
  normalized_state(0, 1) = state(0, 1)/TWO_PI;

  return normalized_state;
}

eig::Array<float, 1, 2> action_normalize(const eig::Array<float, 1, 2>& action, 
                                        const RL::GlobalConfig_t& global_config)
{
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");

  eig::Array<float, 1, 2> normalized_action;
  normalized_action(0, 0) = action(0, 0)/(2.0F*action1_max);
  normalized_action(0, 1) = action(0, 1)/(2.0F*action2_max);

  return normalized_action;
}


template<int BatchSize, 
        int ActorInputSize,  int ... ActorNHiddenLayers, 
        int CriticInputSize, int... CriticNHiddenLayers, 
        typename EigenDerived, typename LossFcn_t, typename LossGradFcn_t>
auto actor_gradient_batch(const ArtificialNeuralNetwork<ActorInputSize, ActorNHiddenLayers ...>&   actor_network,
                          const ArtificialNeuralNetwork<CriticInputSize, CriticNHiddenLayers ...>& critic_network,
                          const eig::ArrayBase<EigenDerived>&                                     input,
                                LossFcn_t&                                                        loss_fcn, 
                                LossGradFcn_t&                                                    loss_grad_fcn)
{
  constexpr int output_len        = ann_output_len<ActorNHiddenLayers ...>::value;
  constexpr int actor_n_layers    = pack_len<ActorInputSize, ActorNHiddenLayers ...>::value;
  constexpr int critic_n_layers   = pack_len<CriticInputSize, CriticNHiddenLayers ...>::value;
  constexpr int largest_layer_len = max_layer_len<ActorInputSize, ActorNHiddenLayers ..., CriticInputSize, CriticNHiddenLayers ...>::value;
  using weight_type   = decltype(actor_network.weight);
  using bias_type     = decltype(actor_network.bias);
  using Delta_t       = eig::Array<float, BatchSize, eig::Dynamic, eig::RowMajor, BatchSize, largest_layer_len>;
  using Activation_t  = eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>;

  tuple<float, weight_type, bias_type> retval = std::make_tuple(0.0F, weight_type{}, bias_type{});
  auto& [loss, weight_grad, bias_grad] = retval;

  // Perform forward propagation
  vector<Activation_t> actor_activations, critic_activations;
  actor_activations.reserve(actor_n_layers);
  critic_activations.reserve(critic_n_layers);
  critic_activations.emplace_back(input);
  actor_activations.emplace_back(input(all, {0, 1}));

  // calculate and store activations at each layer for actor_network
  int weights_count = 0;
  int bias_count    = 0;
  for (int layer = 1; layer <= actor_network.n_layers; layer++)
  {
    int n_nodes_last_layer = actor_network.n_nodes(layer-1);
    int n_nodes_cur_layer  = actor_network.n_nodes(layer);
    int weights_start      = weights_count;
    int weights_end        = weights_start + (n_nodes_cur_layer*n_nodes_last_layer) - 1;
    
    auto b = actor_network.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    actor_activations.emplace_back(BatchSize, n_nodes_cur_layer);
    for (int node = 0; node < n_nodes_cur_layer; node++)
    {
      int this_node_weight_start = weights_start+(node*n_nodes_last_layer);
      int this_node_weight_end   = weights_start+((node+1)*n_nodes_last_layer)-1;
      
      // calculate wX + b
      auto w  = actor_network.weight(seq(this_node_weight_start, this_node_weight_end));
      auto wx = (actor_activations[layer-1].matrix() * w.matrix());
      auto w_temp = w.eval();
      auto wx_temp = wx.eval();
      auto wxb_temp = (wx_temp.array() + b(node)).eval();
      actor_activations[layer](all, node) = actor_network.layer_activation_func[layer-1].activation_batch( (wx.array() + b(node)) );
    }
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }

  // calculate and store activations at each layer for critic_network
  weights_count = 0;
  bias_count    = 0;
  for (int layer = 1; layer <= critic_network.n_layers; layer++)
  {
    int n_nodes_last_layer = critic_network.n_nodes(layer-1);
    int n_nodes_cur_layer  = critic_network.n_nodes(layer);
    int weights_start      = weights_count;
    int weights_end        = weights_start + (n_nodes_cur_layer*n_nodes_last_layer) - 1;
    
    auto b = critic_network.bias(seq(bias_count, bias_count+n_nodes_cur_layer-1));
    critic_activations.emplace_back(BatchSize, n_nodes_cur_layer);
    for (int node = 0; node < n_nodes_cur_layer; node++)
    {
      int this_node_weight_start = weights_start+(node*n_nodes_last_layer);
      int this_node_weight_end   = weights_start+((node+1)*n_nodes_last_layer)-1;
      
      // calculate wX + b
      auto w  = critic_network.weight(seq(this_node_weight_start, this_node_weight_end));
      auto wx = (critic_activations[layer-1].matrix() * w.matrix());
      auto w_temp = w.eval();
      auto wx_temp = wx.eval();
      auto wxb_temp = (wx_temp.array() + b(node)).eval();
      critic_activations[layer](all, node) = critic_network.layer_activation_func[layer-1].activation_batch( (wx.array() + b(node)) );
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
  const int s0      = 0;
  const int s1      = 1;
  const int a0      = 2;
  const int a1      = 3;
  const int r       = 4;
  const int next_s0 = 5;
  const int next_s1 = 6;
  const float discount_factor = 0.9F; // TODO: Tune this

  std::random_device seed;
  std::mt19937 rand_gen(seed());
  std::uniform_real_distribution<float> pose_x_sample(0, world_max_x);
  std::uniform_real_distribution<float> pose_y_sample(0, world_max_y);
  std::uniform_real_distribution<float> heading_sample(-PI, PI);

  constexpr size_t batch_size   = 512u;
  constexpr size_t max_episodes = 100u;
  constexpr size_t replay_buffer_size = 2500u;

  eig::Array<float, eig::Dynamic, 7, eig::RowMajor, replay_buffer_size> replay_buffer;
  size_t replay_buffer_len = 0u;
  size_t episode_count     = 0u; 
  std::normal_distribution<float> exploration_noise(0.0F, 1.0F);

  // Deep deterministic policy gradient
  ArtificialNeuralNetwork<2, 8, 10, 2> sampling_actor, target_actor; 
  ArtificialNeuralNetwork<4, 10, 12, 1> sampling_critic, target_critic;

  sampling_actor.dense(Activation(RELU, HE), 
                       Activation(RELU, HE), 
                       Activation(RELU, HE)
                       );
  // sampling_actor.bias(all, 0) = 0.1F;
  target_actor = sampling_actor;
  
  sampling_critic.dense(Activation(RELU, HE), 
                        Activation(RELU, HE), 
                        Activation(RELU, HE)
                        );
  // sampling_critic.bias(all, 0) = 0.3F;
  target_critic = sampling_critic;

  AdamOptimizer actor_opt((int)sampling_actor.weight.rows(), (int)sampling_actor.bias.rows(), 1e-5F);
  AdamOptimizer critic_opt((int)sampling_critic.weight.rows(), (int)sampling_critic.bias.rows(), 1e-5F);
  
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
      tie(state(0, 0), state(0, 1)) = calc_error(current_state, target_state);
      state = state_normalize(state, global_config);

      action  = forward_batch<1>(sampling_actor, state);
      action += exploration_noise(rand_gen);
      
      // clamp action
      action(0, 0) = std::clamp(action(0, 0), -action1_max, action1_max);
      action(0, 1) = std::clamp(action(0, 1), -action2_max, action2_max);

      // execute the action and observe next state, reward
      auto state_projected = differential_robot(current_state, 
                                                {action(0, 0), action(0, 1)}, 
                                                global_config);
      
      state_projected.psi = RL::wrapto_minuspi_pi(state_projected.psi);

      tie(next_state(0, 0), next_state(0, 1)) = calc_error(state_projected, target_state);
      next_state = state_normalize(next_state, global_config);
      reward = calc_reward(state_projected - target_state);

      // debug code --start
      ENV::update_visualizer({state_projected.x, state_projected.y}, {action(0, 0), action(0, 1)}, reward);
      // debug code --end

      // store current transition -> S_t, A_t, R_t, S_{t+1} in replay buffer
      replay_buffer_len %= replay_buffer_size;
      if (replay_buffer.rows() < (int)(replay_buffer_len+1u))
      { replay_buffer.conservativeResize(replay_buffer_len+1u, NoChange); }
      replay_buffer(replay_buffer_len, {s0, s1})           = state;
      replay_buffer(replay_buffer_len, {a0, a1})           = action_normalize(action, global_config);
      replay_buffer(replay_buffer_len, r)                  = reward;
      replay_buffer(replay_buffer_len, {next_s0, next_s1}) = next_state;
      replay_buffer_len++;

      current_state = state_projected;
      current_cycle++;

      if (current_cycle >= batch_size)
      {
        // Sample 'N' transitions from replay buffer to do mini-batch param optimization
        auto n_transitions = get_n_shuffled_idx<batch_size>((int)replay_buffer.rows());

        eig::Array<float, batch_size, 1> Q_target;
        eig::Array<float, batch_size, 4, eig::RowMajor> target_critic_input;

        target_critic_input(all, {s0, s1}) = replay_buffer(n_transitions, {next_s0, next_s1});
        target_critic_input(all, {a0, a1}) = forward_batch<batch_size>(target_actor, target_critic_input(all, {s0, s1}));

        Q_target  = forward_batch<batch_size>(target_critic, target_critic_input);
        Q_target *= discount_factor;
        Q_target += replay_buffer(n_transitions, r);

        // calculate loss between Q_target, Q_sampling and perform optimization step
        auto [loss_critic, critic_weight_grad, critic_bias_grad] = gradient_batch<batch_size>(sampling_critic, 
                                                                                              replay_buffer(n_transitions, {s0, s1, a0, a1}), 
                                                                                              Q_target, 
                                                                                              critic_loss,
                                                                                              critic_loss_grad<batch_size>);
        critic_opt.step(critic_weight_grad, critic_bias_grad, sampling_critic.weight, sampling_critic.bias);

        // calculate loss for sampling_actor network and perform optimization step
        auto [loss_actor, actor_weight_grad, actor_bias_grad] = actor_gradient_batch<batch_size>(sampling_actor, 
                                                                                                 sampling_critic, 
                                                                                                 replay_buffer(n_transitions, {s0, s1, a0, a1}),
                                                                                                 actor_loss,
                                                                                                 actor_loss_grad);
        actor_opt.step(actor_weight_grad, actor_bias_grad, sampling_actor.weight, sampling_actor.bias);

        // soft-update target networks parameters
        target_actor.weight *= soft_update_rate;
        target_actor.weight += (1.0F - soft_update_rate)*sampling_actor.weight;

        target_critic.bias *= soft_update_rate;
        target_critic.bias += (1.0F - soft_update_rate)*sampling_critic.bias;

        std::cout << "Episode: " << episode_count << ", Cycle: " << current_cycle << ", LossCritic: " << loss_critic << ", LossActor: " << loss_actor << '\n';
      }

      robot_state_inside_world = is_robot_inside_world(current_state, global_config);
    }
    episode_count++;
  }

  return std::make_tuple(target_actor, target_critic);
}

#endif
