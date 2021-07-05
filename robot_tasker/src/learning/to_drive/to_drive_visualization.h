#ifndef _TO_DRIVE_VISUALIZATION_H_
#define _TO_DRIVE_VISUALIZATION_H_

#include <thread>

#include "../../global_typedef.h"
#include "../../ANN/ANN.h"

#include "robot_dynamics.h"
#include "to_drive_util.h"
#include "../../util/util.h"
#include "../../gui/gui.h"
#include "../../gui/gui_util.h"

#include <imgui.h>
#include <imgui-SFML.h>
#include <implot.h>

namespace learning::to_drive
{

template<typename ActorNetwork_t, typename CriticNetwork_t>
void show_me_what_you_learned(const ActorNetwork_t& actor, 
                              const CriticNetwork_t& critic, 
                              const std::string& config_file, 
                              const size_t           max_episodes)
{
  auto global_config  = util::read_global_config(config_file);
  static const float& world_max_x = global_config.at("world/size/x"); 
  static const float& world_max_y = global_config.at("world/size/y"); 
  static const float& action1_max = global_config.at("robot/max_wheel_speed");
  static const float& action2_max = global_config.at("robot/max_wheel_speed");
  static const TargetReachSuccessParams target_reach_params_local = TargetReachSuccessParams{0.5F, 0.5F};

  std::random_device seed;
  std::mt19937 rand_gen(seed());
  std::uniform_real_distribution<float> state_x_sample(0.0F, world_max_x);
  std::uniform_real_distribution<float> state_y_sample(0.0F, world_max_y);
  std::uniform_real_distribution<float> state_psi_sample(-PI, PI);
  std::uniform_real_distribution<float> action1_sample(0.0F, action1_max);
  std::uniform_real_distribution<float> action2_sample(0.0F, action2_max);

  size_t episode_count = 0u;

  std::vector<float> action1_hist;
  std::vector<float> action2_hist;
  std::vector<float> index;
  size_t cycle_count = 0u;

  while ((episode_count < max_episodes))
  {
    episode_count++;

    bool is_episode_done = false;
    DifferentialRobotState cur_state, target_state, next_state;
    eig::Array<float, 1, 3, eig::RowMajor> policy_s_now, policy_s_next;
    eig::Array<float, 1, 2, eig::RowMajor> policy_action;
    eig::Array<float, 1, 5, eig::RowMajor> critic_input;

    tie(std::ignore, target_state) = init_new_episode(state_x_sample, state_y_sample, state_psi_sample, rand_gen);

    if (episode_count == 1u)
    {
      cur_state.x = state_x_sample(rand_gen);
      cur_state.y = state_y_sample(rand_gen);
      cur_state.psi = state_psi_sample(rand_gen);
    }

    gui::set_target_state(target_state.x, target_state.y);

    while (NOT(is_episode_done))
    {
      gui::gui_render_begin();

      tie(policy_s_now(0,0), policy_s_now(0, 1), policy_s_now(0, 2)) = cur_state - target_state;
      state_normalize(global_config, policy_s_now);
      
      policy_action = forward_batch<1>(actor, policy_s_now);
      next_state  = differential_robot(cur_state, {policy_action(0, 0)*action1_max, policy_action(0, 1)*action2_max}, global_config);
      
      critic_input(0, S0) = policy_s_now(0, 0);
      critic_input(0, S1) = policy_s_now(0, 1);
      critic_input(0, S2) = policy_s_now(0, 2);
      critic_input(0, A0) = policy_action(0, 0);
      critic_input(0, A1) = policy_action(0, 1);
      auto Q   = forward_batch<1>(critic, critic_input);

      cur_state = next_state;

      is_episode_done = is_robot_outside_world(next_state, global_config);
      is_episode_done |= has_robot_reached_target(next_state, target_state, target_reach_params_local);

      // draw robot_position
      gui::set_robot_state(next_state.x, next_state.y, next_state.psi);

      // plot action
      if (action1_hist.size() < 50u)
      {
        action1_hist.push_back(policy_action(0, 0)*action1_max);
        action2_hist.push_back(policy_action(0, 1)*action2_max);
        index.push_back((float)cycle_count); 
      }
      else
      {
        int this_index = cycle_count%50u;
        action1_hist[this_index] = policy_action(0, 0)*action1_max;
        action2_hist[this_index] = policy_action(0, 1)*action2_max;
        index[this_index] = (float)cycle_count;
      }

      ImGui::Begin("Testing");
      ImGui::Text("a0");
      if (ImPlot::BeginPlot("##first", NULL, NULL, ImVec2(0, 200), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit))
      {
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 1);
        ImPlot::PlotScatter("", index.data(), action1_hist.data(), (int)action1_hist.size());
        ImPlot::EndPlot();
      }

      ImGui::Text("a1");
      if (ImPlot::BeginPlot("##second", NULL, NULL, ImVec2(0, 200), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit))
      {
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 1);
        ImPlot::PlotScatter("", index.data(), action2_hist.data(), (int)action2_hist.size());
        ImPlot::EndPlot();
      }
      ImGui::End();
      
      gui::gui_render_finalize();
      cycle_count++;
    }
    std::cout << "Reached target\n";
    std::this_thread::sleep_for(1s);

    cycle_count = 0u;
    std::fill(action1_hist.begin(), action1_hist.end(), 0.0F);
    std::fill(action2_hist.begin(), action2_hist.end(), 0.0F);
    std::fill(index.begin(), index.end(), 0.0F);
  }
   std::this_thread::sleep_for(1s);
}


} // namespace {learning::to_drive}

#endif
