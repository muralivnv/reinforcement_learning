#ifndef _TRAINING_VISUALIZER_H_
#define _TRAINING_VISUALIZER_H_

#include "global_typedef.h"
#include "cppyplot.hpp"

void training_visualizer_init()
{
  Cppyplot::cppyplot pyp;
  int temp = 0;
  pyp.raw(R"pyp(
  fig = plt.figure(figsize=(10, 5))
  critic_loss_axes = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
  actor_loss_axes  = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1)
  action_axes      = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1)
  reward_axes      = plt.subplot2grid((3, 2), (0, 1), rowspan=3, colspan=1)

  critic_loss_axes.grid(True)
  critic_loss_axes.set_ylim(-1.0, 1.0)
  critic_loss_axes.set_xlim(0, 100)
  critic_loss_axes.set_ylabel('Critic Loss', fontsize=8)

  actor_loss_axes.grid(True)
  actor_loss_axes.set_ylabel('Actor Loss', fontsize=8)
  actor_loss_axes.set_xlim(0, 100)
  actor_loss_axes.set_ylim(-1.0, 1.0)
  
  action_axes.set_xlim(0, 100)
  action_axes.set_ylim(-5, 5)
  action_axes.set_ylabel("Action", fontsize=8)
  action_axes.tick_params(axis='x', labelsize=6)
  action_axes.tick_params(axis='y', labelsize=6)
  action_axes.grid(True)

  reward_axes.set_xlim(0, 100)
  reward_axes.set_ylim(-100, 20)
  reward_axes.set_xlabel("Cycle", fontsize=8)
  reward_axes.set_ylabel("Reward", fontsize=8)
  reward_axes.tick_params(axis='x', labelsize=6)
  reward_axes.tick_params(axis='y', labelsize=6)
  reward_axes.grid(True)

  fig.tight_layout()

  buffer_idx     = 0
  critic_loss_buffer  = []
  actor_loss_buffer   = []
  action1_buffer      = []
  action2_buffer      = []
  reward_buffer       = []

  critic_loss_plot_obj, = critic_loss_axes.plot([0], [0], 'r--', linewidth=1)
  actor_loss_plot_obj,  = actor_loss_axes.plot([0], [0], 'r--', linewidth=1)
  action1_plot_obj,     = action_axes.plot([0], [0], 'ko--', linewidth=1, markersize=1, alpha=0.8, label="$a_0$")
  action2_plot_obj,     = action_axes.plot([0], [0], 'ro--', linewidth=1, markersize=1, alpha=0.8, label="$a_1$")
  action_axes.legend(loc="upper right")
  reward_plot_obj,      = reward_axes.plot([0], [0], 'm-', linewidth=1)

  fig.canvas.draw()
  fig.canvas.flush_events()

  plt.show(block=False)

  # cache axes background for blitting
  critic_axes_bg  = fig.canvas.copy_from_bbox(critic_loss_axes.bbox)
  actor_axes_bg   = fig.canvas.copy_from_bbox(actor_loss_axes.bbox)
  action_axes_bg = fig.canvas.copy_from_bbox(action_axes.bbox)
  reward_axes_bg = fig.canvas.copy_from_bbox(reward_axes.bbox)
  )pyp", _p(temp));
}


void training_visualizer_update(const float            critic_loss, 
                                const float            actor_loss, 
                                const array<float, 2>& action, 
                                const float            reward)
{
  Cppyplot::cppyplot pyp;

  pyp.raw(R"pyp(
  if (len(action1_buffer) < buffer_idx+1):
    critic_loss_buffer.append(0.0)
    actor_loss_buffer.append(0.0)
    action1_buffer.append(0.0)
    action2_buffer.append(0.0)
    reward_buffer.append(0.0)
  
  critic_loss_buffer[buffer_idx]  = critic_loss
  actor_loss_buffer[buffer_idx]   = actor_loss
  action1_buffer[buffer_idx] = action[0]
  action2_buffer[buffer_idx] = action[1]
  reward_buffer[buffer_idx]  = reward
  buffer_idx                 = (buffer_idx+1)%100
  idx_temp = np.arange(0, len(action1_buffer))

  critic_loss_plot_obj.set_data(idx_temp, 
                                critic_loss_buffer[buffer_idx:] + critic_loss_buffer[:buffer_idx])
  actor_loss_plot_obj.set_data(idx_temp, 
                                actor_loss_buffer[buffer_idx:] + actor_loss_buffer[:buffer_idx])
  action1_plot_obj.set_data(idx_temp, 
                            action1_buffer[buffer_idx:] + action1_buffer[:buffer_idx])
  action2_plot_obj.set_data(idx_temp, 
                            action2_buffer[buffer_idx:] + action2_buffer[:buffer_idx])
  reward_plot_obj.set_data(idx_temp, 
                            reward_buffer[buffer_idx:] + reward_buffer[:buffer_idx])
  
  actor_loss_axes.set_ylim(np.min(actor_loss_buffer), np.max(actor_loss_buffer))
  critic_loss_axes.set_ylim(np.min(critic_loss_buffer), np.max(critic_loss_buffer))
  reward_axes.set_ylim(np.min(reward_buffer)-1.0, np.max(reward_buffer)+1.0)
  
  fig.canvas.restore_region(critic_axes_bg)
  fig.canvas.restore_region(actor_axes_bg)
  fig.canvas.restore_region(action_axes_bg)
  fig.canvas.restore_region(reward_axes_bg)

  critic_loss_axes.draw_artist(critic_loss_plot_obj)
  actor_loss_axes.draw_artist(actor_loss_plot_obj)

  action_axes.draw_artist(action1_plot_obj)
  action_axes.draw_artist(action2_plot_obj)

  reward_axes.draw_artist(reward_plot_obj)

  fig.canvas.blit(critic_loss_axes.bbox)
  fig.canvas.blit(actor_loss_axes.bbox)
  fig.canvas.blit(action_axes.bbox)
  fig.canvas.blit(reward_axes.bbox)
  fig.canvas.flush_events()
  )pyp", _p(critic_loss), _p(actor_loss), _p(action), _p(reward));
}

#endif
