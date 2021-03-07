#ifndef _ENVIRONMENT_UTIL_H_
#define _ENVIRONMENT_UTIL_H_

#include "global_typedef.h"

namespace ENV
{
vector<RL::Polynomial<1>> read_world(const std::string& config_name)
{
  yml::Node config = yml::LoadFile(config_name);
  vector<RL::Polynomial<1>> world_map;

  yml::Node barriers = config["world"]["barriers"];
  for (size_t i = 0u; i < barriers.size(); i++)
  {
    const yml::Node& node = barriers[i];
    const yml::Node& start = node["start"];
    const yml::Node& end   = node["end"];

    unsigned int x1 = start[0].as<unsigned int>();
    unsigned int y1 = start[1].as<unsigned int>();
    unsigned int x2 = end[0].as<unsigned int>();
    unsigned int y2 = end[1].as<unsigned int>();

    if (x1 != x2)
    {
      RL::Polynomial<1> poly;
      poly.coeff[0]  = (float)y1;
      poly.coeff[1]  = (float)(y2 - y1)/(float)(x2 - x1);
      poly.offset    = 0.0F;
      poly.bound_1.x = (float)x1;
      poly.bound_1.y = (float)y1;
      poly.bound_2.x = (float)x2;
      poly.bound_2.y = (float)y2;

      world_map.push_back(poly);
    }
    else
    {
      RL::Polynomial<1> poly;
      poly.coeff[0]  = (float)y1;
      poly.coeff[1]  = INF;
      poly.offset    = 0.0F;
      poly.bound_1.x = (float)x1;
      poly.bound_1.y = (float)y1;
      poly.bound_2.x = (float)x2;
      poly.bound_2.y = (float)y2;

      world_map.push_back(poly);
    }
  }
  return world_map;
}

template<typename ArgLeft, typename ... ArgsRight>
auto min (ArgLeft x, ArgsRight ... y)
{
  typename std::common_type_t<ArgLeft, ArgsRight...> result = x;
  ((result = ((result < y)? result : y) ), ...);
  
  return result;
}

template<typename ArgLeft, typename ... ArgsRight>
auto max (ArgLeft x, ArgsRight ... y)
{
  typename std::common_type_t<ArgLeft, ArgsRight...> result = x;
  ((result = ((result > y)? result : y) ), ...);
  
  return result;
}

template<size_t N>
float eval_poly(RL::Polynomial<N> polynomial, float value)
{
  float result = polynomial.coeff[0];
  float x      = value;
  for (size_t i = 1u; i < (N+1); i++)
  {
    result += (x*polynomial.coeff[i]);
    x *= x;
  }
  return result;
}

template<size_t M, size_t N>
float poly_diff(RL::Polynomial<M> poly1, RL::Polynomial<N> poly2, float value)
{
  float min_x = min(poly1.bound_1.x, poly1.bound_2.x, poly2.bound_1.x, poly2.bound_2.x);
  float max_x = max(poly1.bound_1.x, poly1.bound_2.x, poly2.bound_1.x, poly2.bound_2.x);

  value = std::clamp(value, min_x, max_x);

  return (eval_poly(poly1, value) - eval_poly(poly2, value));
}

bool trajectory_intersects_barrier(const vector<RL::Polynomial<1>> world_map, 
                                   const RL::RobotState            robot_state)
{
  bool path_intersects_with_barrier = false;
  size_t n_cycles = 5u;
  float dt        = 0.04F;
  RL::Array<float, 3> Y; 
  RL::Matrix<float, 3, 3> X;
  float y_ini = robot_state.position.y;
  float x_ini = 0.0F;
  for (size_t i = 0u; i < 3u; i++)
  {
    Y(i, 0) = y_ini + robot_state.velocity.y*dt;
    X(i, 0) = x_ini + robot_state.velocity.x*dt;

    X(i, 1) = X(i, 0)*X(i, 0);
    X(i, 2) = X(i, 1)*X(i, 0);

    y_ini = Y(i, 0);
    x_ini = X(i, 0);
  }
  RL::MatrixX<float> result = (X.inverse() * Y);
  RL::Polynomial<3> robot_path;
  robot_path.coeff[0] = robot_state.position.x;
  robot_path.coeff[1] = result(0, 0);
  robot_path.coeff[2] = result(1, 0);
  robot_path.coeff[3] = result(2, 0);
  robot_path.offset   = robot_state.position.x;

  float interval_x1 = robot_state.position.x;
  float interval_x2 = interval_x1 + X(last, 0);

  for (size_t barrier_iter = 0u; barrier_iter < world_map.size(); barrier_iter++)
  {
    const auto& cur_barrier_poly = world_map[barrier_iter];
    // use bolzano method to determine intersection
    if ((cur_barrier_poly.coeff.size() > 2u) || (cur_barrier_poly.coeff[1] < INF))
    {
      float h1 = poly_diff(cur_barrier_poly, world_map[barrier_iter], interval_x1);
      float h2 = poly_diff(cur_barrier_poly, world_map[barrier_iter], interval_x2);

      if (h1*h2 < 0.0F)
      {
        path_intersects_with_barrier = true;
        break;
      }
    }
    else if (cur_barrier_poly.coeff[1] > (INF-0.1F))
    {
      float h1 = eval_poly(cur_barrier_poly, cur_barrier_poly.bound_1.x);

      if (   ( (h1 > cur_barrier_poly.bound_1.y) && (h1 < cur_barrier_poly.bound_2.y) )
          || ( (h1 < cur_barrier_poly.bound_1.y) && (h1 > cur_barrier_poly.bound_2.y) ) )
      {
        path_intersects_with_barrier = true;
      }
    }
  }

  return path_intersects_with_barrier;
  
}

} // namespace {ENV}

#endif
