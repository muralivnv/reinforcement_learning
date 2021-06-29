#ifndef _GLOBAL_TYPEDEF_H_
#define _GLOBAL_TYPEDEF_H_

#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <string_view>
#include <charconv>
#include <unordered_map>
#include <any>

#include <fstream>
#include <chrono>

#include <algorithm>
#include <numeric>

#pragma warning(push, 0)        
// includes with unfixable warnings
#include <Eigen/Core>
#include <Eigen/Dense>
#include "yaml-cpp/yaml.h"
#pragma warning(pop)

using std::vector, std::string, std::size_t, std::array, std::fstream, std::tuple, std::tie;
using std::for_each, std::copy, std::transform;
using Eigen::all, Eigen::seq, Eigen::last, Eigen::NoChange;

namespace eig    = Eigen;
namespace chrono = std::chrono;
namespace yml    = YAML;
using namespace std::string_literals;
using namespace std::chrono_literals;

#define ITER(X) std::begin(X), std::end(X)
#define TIME_NOW chrono::system_clock::now()
#define TIME_DIFF(end_time, start_time) (std::chrono::duration<float>(end_time - start_time)).count()
#define str2float(str, result) std::from_chars(str.data(), str.data()+str.length(), result)
#define INF 100000.0F
#define UNUSED(X) (void)(X)
#define NOT(X) (!(X))
#define PI (3.14159265F)
#define TWO_PI (2.0F*PI)
#define deg2rad(x) ((x)*0.017453292519943F)
#define rad2deg(x) ((x)*57.2957795130823208F)

namespace RL {

using GlobalConfig_t = std::unordered_map<std::string, float>;

struct State2D{
  float x;
  float y;
};

struct DifferentialRobotState{
  float x;   // (m)
  float y;   // (m)
  float psi; // heading (rad)
};

struct Action2D{
  float action1;
  float action2;
};

template<size_t N>
struct Polynomial{
  eig::Matrix<float, (int)(N+1), 1, eig::ColMajor> coeff;
  float offset;
  State2D bound_1;
  State2D bound_2;
};

struct RobotState{
  State2D position;
  State2D velocity;
};

} // namespace {RL}


#endif
