#ifndef _TO_DRIVE_UTIL_H_
#define _TO_DRIVE_UTIL_H_

#include <cmath>

#include "../../global_typedef.h"

#include "../../ANN/ANN_activation.h"
#include "../../ANN/ANN.h"

#include "../../util/util.h"

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

tuple<float, float> operator-(const RL::DifferentialRobotState& actual, 
                              const RL::DifferentialRobotState& reference);

tuple<RL::DifferentialRobotState, RL::DifferentialRobotState>
init_new_episode(std::uniform_real_distribution<float>& state_x_sample, 
                 std::uniform_real_distribution<float>& state_y_sample, 
                 std::uniform_real_distribution<float>& state_psi_sample, 
                 std::mt19937& rand_gen);

float get_exploration_noise(std::normal_distribution<float>& exploration_noise_dist, 
                           std::mt19937& rand_gen);

void state_normalize(const RL::GlobalConfig_t&               global_config, 
                     eig::Array<float, 1, 2, eig::RowMajor>& policy_state);

bool is_robot_outside_world(const RL::DifferentialRobotState& state,
                            const RL::GlobalConfig_t&         global_config);

bool has_robot_reached_target(const RL::DifferentialRobotState& current_state, 
                              const RL::DifferentialRobotState& target_state, 
                              const TargetReachSuccessParams&   target_reached_criteria);

template<int InputSize, int ... NHiddenLayers>
void soft_update_network(const ANN::ArtificialNeuralNetwork<InputSize, NHiddenLayers...>&  target, 
                         const float smoothing_constant, 
                         ANN::ArtificialNeuralNetwork<InputSize, NHiddenLayers...>&  smoothened_network)
{
  // soft update weights
  const float target_smoothing_factor = 1.0F - smoothing_constant;
  smoothened_network.weight  = (smoothing_constant*smoothened_network.weight) + (target_smoothing_factor*target.weight);

  // soft update bias
  smoothened_network.bias  = (smoothing_constant*smoothened_network.bias) + (target_smoothing_factor*target.bias);
}

template<int BatchSize, 
        int ActorInputSize,  int ... ActorNHiddenLayers, 
        int CriticInputSize, int... CriticNHiddenLayers, 
        typename EigenDerived, typename LossFcn_t, typename LossGradFcn_t>
auto actor_gradient_batch(const ANN::ArtificialNeuralNetwork<ActorInputSize, ActorNHiddenLayers ...>&   actor_network,
                          const ANN::ArtificialNeuralNetwork<CriticInputSize, CriticNHiddenLayers ...>& critic_network,
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
  critic_activations[0](all, {S0, S1}) = input;
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
      actor_activations[layer](all, node) = actor_network.layer_activation_func[layer-1].activation_batch(z);
    }
    weights_count  += n_nodes_last_layer*n_nodes_cur_layer;
    bias_count     += n_nodes_cur_layer;
  }

  // clamp actions
  critic_activations[0](all, {A0, A1}) = actor_activations.back();

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
      critic_activations[layer](all, node) = critic_network.layer_activation_func[layer-1].activation_batch(z);
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
  delta_to_here = delta(all, {A0, A1}); // only take gradient with respect to action_0 and action_1
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

#endif
