#ifndef _ENVIRONMENT_UTIL_H_
#define _ENVIRONMENT_UTIL_H_

#include <filesystem> 
#include <fstream>
#include <unordered_map>

#include "../global_typedef.h"
#include "util.h"
#include "cppyplot.hpp"

namespace filesystem = std::filesystem;

namespace env_util
{

std::unordered_map<std::string, float>
read_global_config(const std::string& config_name)
{
  std::unordered_map<std::string, float> retval;
  yml::Node config = yml::LoadFile(config_name);

  yml::Node node = config["world"]["size"];
  retval.insert(std::make_pair("world/size/x", node[0].as<float>()));
  retval.insert(std::make_pair("world/size/y", node[0].as<float>()));

  yml::Node robot_config = config["robot"];
  retval.insert(std::make_pair("robot/max_wheel_speed", robot_config["max_wheel_speed"].as<float>()));
  retval.insert(std::make_pair("robot/wheel_radius", robot_config["wheel_radius"].as<float>()));
  retval.insert(std::make_pair("robot/base_length", robot_config["base_length"].as<float>()));

  retval.insert(std::make_pair("cycle_time", config["cycle_time"].as<float>()));
  
  return retval;
}

void initiate_new_world(std::string_view file_to_save)
{
  Cppyplot::cppyplot pyp;
  pyp.raw(R"pyp(
  global_params = open('/'.join(file_to_save.split('/')[:-1])+'/global_params.yaml', 'r')
  params = yaml.load(global_params)
  world_size_x = params['world']['size'][0]
  world_size_y = params['world']['size'][1]
  global_params.close()
  
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  line, = ax.plot([], [], 'rx', markersize=2)
  plt.xlim(0, world_size_x)
  plt.ylim(0, world_size_y)
  plt.title("Draw barriers", fontsize=14)
  plt.xlabel("X(m)", fontsize=12)
  plt.ylabel("Y(m)", fontsize=12)
  paint_data = {}
  paint_data['left_mouse_clicked'] = False
  paint_data['mouse_coords']       = []
  paint_data['poly_fit']           = []
  paint_data['polyfit_eval']       = []

  def on_mouse_move(event):
    if paint_data['left_mouse_clicked']:
      x, y = event.xdata, event.ydata
      paint_data['mouse_coords'].append([x, y])

  def on_left_mouse_click(event):
    if (event.button is MouseButton.LEFT):
      x, y = event.xdata, event.ydata
      paint_data['left_mouse_clicked'] = True
      paint_data['mouse_coords'].append([x, y])

    elif (event.button is MouseButton.RIGHT):
      paint_data['left_mouse_clicked'] = False
      
      final_data = np.array(paint_data['mouse_coords'])
      poly_coeff = np.polyfit(final_data[:, 0], final_data[:, 1], 3)
      #if (poly_coeff[-1] > 10000) or (poly_coeff[-1] < -10000):
      #  return
      polyfit_eval = np.poly1d(poly_coeff)(final_data[:, 0])

      paint_data['mouse_coords'] = []
      paint_data['polyfit_eval'].append([final_data[:, 0], polyfit_eval])
      paint_data['poly_fit'].append([final_data[0, 0], final_data[0, 1], final_data[-1, 0], final_data[-1, 1], 
                                      poly_coeff[0], poly_coeff[1], poly_coeff[2], poly_coeff[3]])

  def on_fig_closed(event):
    np.savetxt(file_to_save, 
                  np.array(paint_data['poly_fit']), fmt='%5.6f', delimiter=',', comments='#',
                  header="x_left,y_left,x_right,y_right,c3,c2,c1,c0")

  def draw_init():
    line.set_data([], [])
    return (line, )
  
  def draw_update(frame):
    line_seq = []
    if (any(paint_data['mouse_coords'])):
      data = np.array(paint_data['mouse_coords'])
      line.set_data(data[:, 0], data[:, 1])
    line_seq.append(line)

    for i in range(0, len(paint_data['polyfit_eval'])):
      new_line, = ax.plot(paint_data['polyfit_eval'][i][0], paint_data['polyfit_eval'][i][1], 'k-')
      line_seq.append(new_line)
    return line_seq
  
  plt.connect('motion_notify_event',  on_mouse_move)
  plt.connect('button_press_event',   on_left_mouse_click)
  plt.connect('close_event',          on_fig_closed)

  anim = FuncAnimation(fig, draw_update, init_func=draw_init, 
                      frames=1000, interval=2, blit=True)
  plt.show()
  )pyp", _p(file_to_save));
}

void realtime_visualizer_init(std::string_view config, int ring_buffer_len)
{
  Cppyplot::cppyplot pyp;
  pyp.raw(R"pyp(
  global_params = open(config, 'r')
  params = yaml.load(global_params, Loader=yaml.SafeLoader)
  world_size_x = params['world']['size'][0]
  world_size_y = params['world']['size'][1]
  global_params.close()

  fig = plt.figure(figsize=(10, 5))
  world_axes    = plt.subplot2grid((3,3), (0,0), rowspan=3, colspan=2)
  action_axes   = plt.subplot2grid((3,3), (0,2), rowspan=2)
  Q_axes   = plt.subplot2grid((3,3), (2,2), colspan=1)

  world_axes.set(facecolor='#f2f4f4')
  world_axes.set_xlim(0, world_size_x)
  world_axes.set_ylim(0, world_size_y)
  world_axes.set_xlabel("X (m)", fontsize=12)
  world_axes.set_ylabel("Y (m)", fontsize=12)
  
  action_axes.set_xlim(0, ring_buffer_len)
  action_axes.set_ylim(-5, 5)
  action_axes.set_ylabel("Action", fontsize=8)
  action_axes.tick_params(axis='x', labelsize=6)
  action_axes.tick_params(axis='y', labelsize=6)
  action_axes.grid(True)

  Q_axes.set_xlim(0, ring_buffer_len)
  Q_axes.set_ylim(-20.0, 5.0)
  Q_axes.set_xlabel("Cycle", fontsize=8)
  Q_axes.set_ylabel("Reward", fontsize=8)
  Q_axes.tick_params(axis='x', labelsize=6)
  Q_axes.tick_params(axis='y', labelsize=6)
  Q_axes.grid(True)

  fig.tight_layout()

  buffer_idx     = 0
  pose_x_buffer  = []
  pose_y_buffer  = []
  action1_buffer = []
  action2_buffer = []
  Q_buffer  = []

  robot_pose_plot_obj, = world_axes.plot(0, 0, 'bx', markersize=8, alpha=0.8)
  target_pose_plot_obj, = world_axes.plot(0, 0, 'ro', markersize=12, alpha=0.9)
  robot_traj_plot_obj, = world_axes.plot(0, 0, 'r--', linewidth=0.8, alpha=0.6)
  action1_plot_obj,    = action_axes.plot([0], [0], 'ko--', linewidth=1, markersize=1, alpha=0.8, label="$a_0$")
  action2_plot_obj,    = action_axes.plot([0], [0], 'ro--', linewidth=1, markersize=1, alpha=0.8, label="$a_1$")
  action_axes.legend(loc="upper right")
  Q_plot_obj,     = Q_axes.plot([0], [0], 'm-', linewidth=1)

  fig.canvas.draw()
  fig.canvas.flush_events()

  plt.show(block=False)

  # cache axes background for blitting
  world_axes_bg  = fig.canvas.copy_from_bbox(world_axes.bbox)
  action_axes_bg = fig.canvas.copy_from_bbox(action_axes.bbox)
  Q_axes_bg = fig.canvas.copy_from_bbox(Q_axes.bbox)

  )pyp", _p(config), _p(ring_buffer_len));
}

void update_target_pose(const array<float, 2>& target_pose)
{
  Cppyplot::cppyplot pyp;
  pyp.raw(R"pyp(
  target_pose_plot_obj.set_data(target_pose[0], target_pose[1])
  
  fig.canvas.restore_region(world_axes_bg)
  fig.canvas.restore_region(action_axes_bg)
  fig.canvas.restore_region(Q_axes_bg)

  world_axes.draw_artist(robot_pose_plot_obj)
  world_axes.draw_artist(target_pose_plot_obj)
  world_axes.draw_artist(robot_traj_plot_obj)

  action_axes.draw_artist(action1_plot_obj)
  action_axes.draw_artist(action2_plot_obj)

  Q_axes.draw_artist(Q_plot_obj)

  fig.canvas.blit(world_axes.bbox)
  fig.canvas.blit(action_axes.bbox)
  fig.canvas.blit(Q_axes.bbox)
  fig.canvas.flush_events()
  )pyp", _p(target_pose));
}

void update_visualizer(const array<float, 2>& pose, 
                       const array<float, 2>& action, 
                       const float            Q,
                       const int ring_buffer_len)
{
  Cppyplot::cppyplot pyp;

  pyp.raw(R"pyp(
  if (len(action1_buffer) < buffer_idx+1):
    pose_x_buffer.append(0.0)
    pose_y_buffer.append(0.0)
    action1_buffer.append(0.0)
    action2_buffer.append(0.0)
    Q_buffer.append(0.0)
  
  pose_x_buffer[buffer_idx]  = pose[0]
  pose_y_buffer[buffer_idx]  = pose[1]
  action1_buffer[buffer_idx] = action[0]
  action2_buffer[buffer_idx] = action[1]
  Q_buffer[buffer_idx]  = Q
  buffer_idx                 = (buffer_idx+1)%ring_buffer_len
  idx_temp = np.arange(0, len(action1_buffer))

  robot_pose_plot_obj.set_data(pose[0], pose[1])
  robot_traj_plot_obj.set_data(pose_x_buffer[buffer_idx:] + pose_x_buffer[:buffer_idx], 
                               pose_y_buffer[buffer_idx:] + pose_y_buffer[:buffer_idx])
  action1_plot_obj.set_data(idx_temp, 
                            action1_buffer[buffer_idx:] + action1_buffer[:buffer_idx])
  action2_plot_obj.set_data(idx_temp, 
                            action2_buffer[buffer_idx:] + action2_buffer[:buffer_idx])
  Q_plot_obj.set_data(idx_temp, 
                            Q_buffer[buffer_idx:] + Q_buffer[:buffer_idx])
  
  fig.canvas.restore_region(world_axes_bg)
  fig.canvas.restore_region(action_axes_bg)
  fig.canvas.restore_region(Q_axes_bg)

  world_axes.draw_artist(robot_pose_plot_obj)
  world_axes.draw_artist(target_pose_plot_obj)
  world_axes.draw_artist(robot_traj_plot_obj)

  action_axes.draw_artist(action1_plot_obj)
  action_axes.draw_artist(action2_plot_obj)

  Q_axes.draw_artist(Q_plot_obj)

  fig.canvas.blit(world_axes.bbox)
  fig.canvas.blit(action_axes.bbox)
  fig.canvas.blit(Q_axes.bbox)
  fig.canvas.flush_events()

  )pyp", _p(pose), _p(action), _p(Q), _p(ring_buffer_len));
}

} // namespace {env_util}

#endif
