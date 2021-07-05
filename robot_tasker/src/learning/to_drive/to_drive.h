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

#include <imgui.h>
#include <imgui-SFML.h>
#include <implot.h>
#include "../../gui/gui.h"
#include "../../gui/gui_util.h"

using namespace ANN;

namespace learning::to_drive
{

#define N_STATES  (3)
#define N_ACTIONS (2)

// local parameters
static const TargetReachSuccessParams target_reach_params = TargetReachSuccessParams{0.5F, 0.5F, deg2rad(5.0F)};

// reward calculation parameters
// normalized x error reward calculation
static const float normalized_x_error_reward_interp_x1 = 0.01F;
static const float normalized_x_error_reward_interp_y1 = -0.1F;
static const float normalized_x_error_reward_interp_x2 = 0.80F;
static float normalized_x_error_reward_interp_y2       = -2.0F; // TODO: make this const after tuning is done

// normalized y error reward calculation
static const float normalized_y_error_reward_interp_x1 = 0.01F;
static const float normalized_y_error_reward_interp_y1 = -0.1F;
static const float normalized_y_error_reward_interp_x2 = 0.80F;
static float normalized_y_error_reward_interp_y2       = -2.0F; // TODO: make this const after tuning is done

// normalized heading error reward calculation
static const float normalized_heading_error_reward_interp_x1 = 0.01F;
static const float normalized_heading_error_reward_interp_y1 = -0.1F;
static const float normalized_heading_error_reward_interp_x2 = 0.80F;
static float normalized_heading_error_reward_interp_y2       = -4.0F; // TODO: make this const after tuning is done

// reward discount factor
static float discount_factor = 0.2F; // TODO: make this const after tuning is done

// function definitions
float calc_reward(eig::Array<float, 1, N_STATES, eig::RowMajor>& normalized_policy_state)
{
  // calculate reward for x error
  float reward = util::linear_interpolate(fabsf(normalized_policy_state(0, S0)), 
                                        normalized_x_error_reward_interp_x1,  normalized_x_error_reward_interp_y1,
                                        normalized_x_error_reward_interp_x2,  normalized_x_error_reward_interp_y2);
  // calculate reward for y error
  reward += util::linear_interpolate(fabsf(normalized_policy_state(0, S1)),   
                                   normalized_y_error_reward_interp_x1,  normalized_y_error_reward_interp_y1,
                                   normalized_y_error_reward_interp_x2,  normalized_y_error_reward_interp_y2);

  reward += util::linear_interpolate(fabsf(normalized_policy_state(0, S2)),   
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


auto learn_to_drive(const learning::to_drive::global_config_t& global_config, 
                    const bool logging_enabled = true)
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
  float  actor_l2_reg_factor = 1e-2F;
  float  critic_l2_reg_factor = 1e-3F;

  // experience replay setup
  eig::Array<float, eig::Dynamic, BUFFER_LEN, eig::RowMajor> replay_buffer;

  // function approximation setup
  ArtificialNeuralNetwork<N_STATES, 8, 12, 20, 25, N_ACTIONS> actor;
  ArtificialNeuralNetwork<N_STATES+N_ACTIONS, 10, 14, 21, 27, 1> critic, critic_target;
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
  std::vector<float> actor_loss_hist;
  std::vector<float> critic_loss_hist;
  std::vector<float> actor_loss_index;
  std::vector<float> critic_loss_index;

  bool terminate_actor_optim = false;
  bool terminate_critic_optim = false;

  // variables for training during test
  DifferentialRobotState cur_state_test, next_state_test, target_state_test;
  eig::Array<float, 1, N_STATES, eig::RowMajor> policy_s_now_test;
  eig::Array<float, 1, N_ACTIONS, eig::RowMajor> policy_action_test;
  std::vector<float> x_error_test, y_error_test, psi_error_test, index_test;
  size_t test_cycle_count = 0u;
  bool initialized_test = false;

  while(   (episode_count < max_episodes        ) 
        && (   (terminate_actor_optim == false ) 
            || (terminate_critic_optim == false)) )
  {
    DifferentialRobotState cur_state, next_state, target_state;
    eig::Array<float, 1, N_STATES, eig::RowMajor> policy_s_now, policy_s_next;
    eig::Array<float, 1, N_ACTIONS, eig::RowMajor> policy_action;
    float reward;
    bool episode_done = false;

    tie(cur_state, target_state) = init_new_episode(state_x_sample, state_y_sample, state_psi_sample, rand_gen);

    while(NOT(episode_done))
    {
      gui::gui_render_begin();

      // add necessary sliders and buttons
      ImGui::Begin("LearningToDrive");
      ImGui::Text("TuningParams");
      ImGui::SliderFloat("discount", &discount_factor, 0.0F, 1.0F);
      ImGui::SliderFloat("max_x_error", &normalized_x_error_reward_interp_y2, -1.0F, -25.0F);
      ImGui::SliderFloat("max_y_error", &normalized_y_error_reward_interp_y2, -1.0F, -25.0F);
      ImGui::SliderFloat("max_psi_error", &normalized_heading_error_reward_interp_y2, -1.0F, -25.0F);
      ImGui::SliderFloat("actor_l2_reg", &actor_l2_reg_factor, 1e-08F, 1e-01F);
      ImGui::SliderFloat("critic_l2_reg", &critic_l2_reg_factor, 1e-08F, 1e-01F);
      ImGui::End();

      // sample random robot state from uniform distribution (for better exploration)
      cur_state.x = state_x_sample(rand_gen);
      cur_state.y = state_y_sample(rand_gen);
      cur_state.psi = state_psi_sample(rand_gen);

      target_state.x = state_x_sample(rand_gen);
      target_state.y = state_y_sample(rand_gen);
      target_state.psi = state_psi_sample(rand_gen);
      
      // calculate policy state
      tie(policy_s_now(0, 0), policy_s_now(0, 1), policy_s_now(0, 2)) = cur_state - target_state;
      state_normalize(global_config, policy_s_now);

      // sample random robot state from uniform distribution (for better exploration)
      policy_action(0, 0) = action1_sample(rand_gen)/action1_max;
      policy_action(0, 1) = action2_sample(rand_gen)/action2_max;

      // perturb system with sample state and actions to observe next state and reward
      next_state  = differential_robot(cur_state, {policy_action(0, 0)*action1_max, policy_action(0, 1)*action2_max}, global_config);
      next_state.psi = util::wrapto_minuspi_pi(next_state.psi);

      tie(policy_s_next(0, 0), policy_s_next(0, 1), policy_s_next(0, 2)) = next_state - target_state;
      state_normalize(global_config, policy_s_next);
      reward = calc_reward(policy_s_next);

      // check whether a reset is required or not
      episode_done = is_robot_outside_world(next_state, global_config);
      episode_done |= has_robot_reached_target(next_state, target_state, target_reach_params);

      // store current transition -> S, A, R, S in replay buffer
      replay_buffer_len %= replay_buffer_size;
      if (replay_buffer.rows() < ((int)replay_buffer_len+1u) )
      { replay_buffer.conservativeResize(replay_buffer_len+1u, NoChange); }

      replay_buffer(replay_buffer_len, {S0, S1, S2}) = policy_s_now;
      replay_buffer(replay_buffer_len, {A0, A1}) = policy_action;
      replay_buffer(replay_buffer_len, R) = reward;
      replay_buffer(replay_buffer_len, {NEXT_S0, NEXT_S1, NEXT_S2}) = policy_s_next;
      replay_buffer(replay_buffer_len, EPISODE_STATE) = (episode_done == true)?0.0F:1.0F;
      replay_buffer_len++;

      if (cycle_count >= warm_up_cycles)
      {
        // sample n measurements of length batch_size
        auto n_transitions = util::get_n_shuffled_indices<batch_size>((int)replay_buffer.rows());

        const float actor_loss_prev = actor_loss_avg;
        const float critic_loss_prev = critic_loss_avg;
        float actor_wgrad_norm  = 0.0F;
        float critic_wgrad_norm = 0.0F;

        if (NOT(terminate_critic_optim))
        {
          eig::Array<float, batch_size, N_STATES + N_ACTIONS, eig::RowMajor> critic_next_input;
          eig::Array<float, batch_size, 1> Q_next, Q_now, td_error;

          // propagate next_state through actor network to calculate A_next = mu(S_next)
          critic_next_input(all, {S0, S1, S2}) = replay_buffer(n_transitions, {NEXT_S0, NEXT_S1, NEXT_S2});
          critic_next_input(all, {A0, A1}) = forward_batch<batch_size>(actor, critic_next_input(all, {S0, S1, S2}));

          // use citic network with A_next, S_next to calculate Q(S_next, A_next)
          Q_next = forward_batch<batch_size>(critic_target, critic_next_input);

          // use critic network with A_now, S_now to calculate Q(S_now, A_now)
          Q_now = forward_batch<batch_size>(critic, replay_buffer(n_transitions, {S0, S1, S2, A0, A1}));

          // calculate temporal difference = R + gamma*Q(S_next, A_next) - Q(S_now, A_now)
          td_error = replay_buffer(n_transitions, (int)R) + (discount_factor*replay_buffer(n_transitions, (int)EPISODE_STATE)*Q_next) - Q_now;

          // calculate gradient of critic network
          auto [critic_loss, critic_weight_grad, critic_bias_grad] = gradient_batch<batch_size>(critic, 
                                                                                                replay_buffer(n_transitions, {S0, S1, S2, A0, A1}),
                                                                                                td_error,
                                                                                                critic_loss_fcn,
                                                                                                critic_loss_grad);
          critic_weight_grad += (critic_l2_reg_factor*critic.weight)/(float)batch_size;

          // update parameters of critic using optimizer
          critic_opt.step(critic_weight_grad, critic_bias_grad, critic.weight, critic.bias);
          critic_wgrad_norm = std::sqrtf(critic_weight_grad.square().sum());

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

          if (critic_loss_hist.size() < 1000)
          { 
            critic_loss_hist.push_back(critic_loss_avg); 
            critic_loss_index.push_back((float)cycle_count);
          }
          else
          {
            auto index = cycle_count % 1000;
            critic_loss_hist[index] = critic_loss_avg;
            critic_loss_index[index] = (float)cycle_count;
          }
        }

        if (NOT(terminate_actor_optim))
        {
          // calculate gradient of actor network
          auto [actor_loss, actor_weight_grad, actor_bias_grad] = actor_gradient_batch<batch_size>(actor, 
                                                                                                  critic, 
                                                                                                  replay_buffer(n_transitions, {S0, S1, S2}),
                                                                                                  actor_loss_fcn,
                                                                                                  actor_loss_grad, 
                                                                                                  global_config);
          // Use L2 regularization of weights for both actor and critic network
          actor_weight_grad  += (actor_l2_reg_factor*actor.weight)/(float)batch_size;

          // update parameters of actor using optimizer
          actor_opt.step(actor_weight_grad, actor_bias_grad, actor.weight, actor.bias);
          actor_wgrad_norm = std::sqrtf(actor_weight_grad.square().sum());

          // update average loss
          actor_loss_avg *= loss_smoothing_factor;
          actor_loss_avg += (1.0F - loss_smoothing_factor)*actor_loss;

          if (actor_loss_hist.size() < 1000)
          {
            actor_loss_hist.push_back(actor_loss_avg); 
            actor_loss_index.push_back((float)cycle_count);
          }
          else
          {
            auto index = cycle_count % 1000;
            actor_loss_hist[index] = actor_loss_avg;
            actor_loss_index[index] = (float)cycle_count;
          }
        }

        if (logging_enabled == true)
        {
          if ( (cycle_count % 20) == 0)
          {
            std::cout << "epoch: " << cycle_count
                      << " | loss_actor: " << actor_loss_avg << " | loss_critic: " << critic_loss_avg
                      << " | actor_wgradNorm: " << actor_wgrad_norm
                      << " | critic_wgradNorm: " << critic_wgrad_norm << '\n';
            std::cout << std::flush;
          }
        }

        ImGui::Begin("LearningToDrive");

        ImGui::Text("Training");
        static bool run_actor_optim = !terminate_actor_optim;
        ImGui::Checkbox("actor_optim", &run_actor_optim);
        terminate_actor_optim = !run_actor_optim;

        static bool run_critic_optim = !terminate_critic_optim;
        ImGui::Checkbox("critic_optim", &run_critic_optim);
        terminate_critic_optim = !run_critic_optim;

        if (ImPlot::BeginPlot("actor_loss", NULL, NULL, ImVec2(0, 200),  ImPlotFlags_CanvasOnly, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit))
        {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 1);
          ImPlot::PlotScatter("", actor_loss_index.data(), actor_loss_hist.data(), (int)actor_loss_hist.size());
          ImPlot::EndPlot();
        }

        if (ImPlot::BeginPlot("critic_loss", NULL, NULL, ImVec2(0, 200), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit))
        {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 1);
          ImPlot::PlotScatter("", critic_loss_index.data(), critic_loss_hist.data(), (int)critic_loss_hist.size());
          ImPlot::EndPlot();
        }
        ImGui::End();

        // ImGui::Begin("GeneratedSamples");
        // ImGui::Text("State_y vs State_x");
        // if (ImPlot::BeginPlot("State_y vs State_x", NULL, NULL, ImVec2(0, 200), ImPlotFlags_CanvasOnly))
        // {
        //   if (replay_buffer.rows() >= 5000)
        //   {
        //     eig::Array<float, -1, 1> state_x(5000, 1), state_y(5000, 1);
        //     state_x(all, 0) = replay_buffer(seq(0, 4999), (int)S0);
        //     state_y(all, 0) = replay_buffer(seq(0, 4999), (int)S1);

        //     ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 1);
        //     ImPlot::SetNextPlotLimits(-1.0F, 1.0F, -1.0F, 1.0F, ImGuiCond_Always);
        //     ImPlot::PlotScatter("", state_x.data(), state_y.data(), (int)state_x.size());
        //     ImPlot::EndPlot();
        //   }
        // }
        // ImGui::Text("a_1 vs a_0");
        // if (ImPlot::BeginPlot("a_1 vs a_0", NULL, NULL, ImVec2(0, 200), ImPlotFlags_CanvasOnly))
        // {
        //   if (replay_buffer.rows() >= 5000)
        //   {
        //     eig::Array<float, -1, 1> a_0(5000, 1), a_1(5000, 1);
        //     a_0(all, 0) = replay_buffer(seq(0, 4999), (int)A0);
        //     a_1(all, 0) = replay_buffer(seq(0, 4999), (int)A1);

        //     ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 1);
        //     ImPlot::SetNextPlotLimits(0.0F, 1.0F, 0.0F, 1.0F, ImGuiCond_Always);
        //     ImPlot::PlotScatter("", a_0.data(), a_1.data(), (int)a_0.size());
        //     ImPlot::EndPlot();
        //   }
        // }
        // ImGui::End();

        if ( (terminate_critic_optim == true) && (terminate_actor_optim == true) )
        { break; }
      }
      cycle_count   = util::min(++cycle_count, std::numeric_limits<size_t>::max());

      // test during training
      {
        if (initialized_test == false)
        {
          tie(cur_state_test, target_state_test) = init_new_episode(state_x_sample, state_y_sample, state_psi_sample, rand_gen);
          gui::set_target_state(target_state_test.x, target_state_test.y);

          initialized_test = true;
          test_cycle_count = 0u;
          
          std::fill(x_error_test.begin(), x_error_test.end(), 0.0F);
          std::fill(y_error_test.begin(), y_error_test.end(), 0.0F);
          std::fill(psi_error_test.begin(), psi_error_test.end(), 0.0F);
          std::fill(index_test.begin(), index_test.end(), 0.0F);
        }

        tie(policy_s_now_test(0, 0), policy_s_now_test(0, 1), policy_s_now_test(0, 2)) = cur_state_test - target_state_test;
        state_normalize(global_config, policy_s_now_test);

        policy_action_test = forward_batch<1>(actor, policy_s_now_test);
        next_state_test  = differential_robot(cur_state_test, {policy_action_test(0, 0)*action1_max, policy_action_test(0, 1)*action2_max}, global_config);
        next_state_test.psi = util::wrapto_minuspi_pi(next_state_test.psi);

        gui::set_robot_state(next_state_test.x, next_state_test.y, next_state_test.psi);
        cur_state_test = next_state_test;

        // check whether a reset is required or not
        bool test_episode_done = is_robot_outside_world(next_state_test, global_config);
        test_episode_done |= has_robot_reached_target(next_state_test, target_state_test, target_reach_params);

        initialized_test = !test_episode_done;

        if (x_error_test.size() < 1000u)
        {
          auto [x_error, y_error, psi_error] = next_state_test - target_state_test;
          x_error_test.push_back(x_error);
          y_error_test.push_back(y_error);
          psi_error_test.push_back(rad2deg(psi_error));
          index_test.push_back((float)test_cycle_count);
        }
        else
        {
          auto [x_error, y_error, psi_error] = next_state_test - target_state_test;
          int index = test_cycle_count % 1000;
          x_error_test[index] = x_error;
          y_error_test[index] = y_error;
          psi_error_test[index] = rad2deg(psi_error);
          index_test[index] = (float)test_cycle_count;
        }

        ImGui::Begin("TestError");
        ImGui::Text("x_error");
        if (ImPlot::BeginPlot("x_error", NULL, NULL, ImVec2(0, 200), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit))
        {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 1);
          ImPlot::PlotScatter("", index_test.data(), x_error_test.data(), (int)index_test.size());
          ImPlot::EndPlot();
        }
        ImGui::Text("y_error");
        if (ImPlot::BeginPlot("y_error", NULL, NULL, ImVec2(0, 200), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit))
        {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 1);
          ImPlot::PlotScatter("", index_test.data(), y_error_test.data(), (int)index_test.size());
          ImPlot::EndPlot();
        }
        ImGui::Text("psi_error");
        if (ImPlot::BeginPlot("psi_error", NULL, NULL, ImVec2(0, 200), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit))
        {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 1);
          ImPlot::PlotScatter("", index_test.data(), psi_error_test.data(), (int)index_test.size());
          ImPlot::EndPlot();
        }
        ImGui::End();

        test_cycle_count++;
      }

      gui::gui_render_finalize();
    }

    episode_count++;
  }

  std::cout << "I know how to Drive!\n";
  return std::make_tuple(actor, critic);
}

} // namespace {learning::to_drive}

#endif