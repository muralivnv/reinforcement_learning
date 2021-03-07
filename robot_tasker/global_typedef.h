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
using Eigen::all, Eigen::seq, Eigen::last;

namespace eig    = Eigen;
namespace chrono = std::chrono;
namespace yml    = YAML;
using namespace std::string_literals;

#define ITER(X) std::begin(X), std::end(X)
#define TIME_NOW chrono::system_clock::now()
#define str2float(str, result) std::from_chars(str.data(), str.data()+str.length(), result)
#define INF 100000.0F
#define UNUSED(X) (void)(X)

namespace RL {
template<typename T>
using MatrixX = eig::Matrix<T, eig::Dynamic, eig::Dynamic, eig::RowMajor>; 

template<typename T, int Rows, int Cols>
using Matrix = eig::Matrix<T, Rows, Cols, eig::RowMajor>;

template<typename T, int Rows>
using Array = eig::Matrix<T, Rows, 1, eig::ColMajor>;

template<int Rows>
using Arrayf = eig::Matrix<float, Rows, 1, eig::ColMajor>;

template<typename T>
using VectorX = eig::Matrix<T, eig::Dynamic, 1, eig::ColMajor>;

struct State2D{
  float x;
  float y;
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
